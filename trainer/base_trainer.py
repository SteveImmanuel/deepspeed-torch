import time
import os
import torch as T
import yaml
import deepspeed
from abc import abstractmethod, ABC
from deepspeed import DeepSpeedEngine
from dataclasses import dataclass
from typing import Dict
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from constants import *
from utils.logger import get_logger

@dataclass
class Tracker:
    last_loss: float = None
    last_metric: float  = None
    epoch: int = 0
    step_counter: int = 0
    val_step_counter: int = 0
    best_epoch: int = None
    best_metric: float = None
    direction: str = 'max'

    def inc_step_counter(self):
        self.step_counter += 1
    
    def inc_val_step_counter(self):
        self.val_step_counter += 1
    
    def is_metric_better(self, epoch: int) -> bool:
        def _compare(a, b):
            return a > b if self.direction == 'max' else a < b

        if self.best_metric is None or _compare(self.last_metric, self.best_metric):
            self.best_metric = self.last_metric
            self.best_epoch = epoch
            return True
        return False
    
    def to_dict(self):
        return {
            'last_loss': self.last_loss,
            'last_metric': self.last_metric,
            'epoch': self.epoch,
            'step_counter': self.step_counter,
            'val_step_counter': self.val_step_counter,
            'best_epoch': self.best_epoch,
            'best_metric': self.best_metric,
            'direction': self.direction
        }

class BaseTrainer(ABC):
    def __init__(self, model: T.nn.Module, local_rank: int, global_rank: int, args: Dict, log_enabled: bool = True, is_eval: bool = False):
        self.logger = get_logger(__class__.__name__) if self.logger is None else self.logger
        self.model = model
        self.args = args
        self.local_rank = local_rank
        self.global_rank = global_rank
        self.log_enabled = log_enabled
        self.is_eval = is_eval
        
        self.uid = args['train']['uid'] if args['train']['uid'] is not None else int(time.time())
        args['train']['uid'] = self.uid
        self.loss_fn = self._get_loss_fn()

        self.optim = self._get_optimizer() if not is_eval else None
        self.scheduler = self._get_scheduler()  if not is_eval else None

        if self.can_log:
            self.log_dir = os.path.join(args['train']['log_dir'], f'{self.uid}')
            self.summary_writer = SummaryWriter(log_dir=self.log_dir)
            self.ckpt_dir = os.path.join(self.log_dir, 'weights')
            
            os.makedirs(self.ckpt_dir, exist_ok=True)
            self.save_config()

        self.tracker = Tracker()
        self.model, self.optim, _, self.scheduler = deepspeed.initialize(
            config=args['train']['deepspeed'],
            model=self.model,
            model_parameters=model.parameters(), 
            optimizer=self.optim,
            lr_scheduler=self.scheduler
        )
        self.model: DeepSpeedEngine
    
    @property
    def is_main_process(self):
        return self.global_rank == 0

    @property
    def can_log(self):
        return self.log_enabled and self.is_main_process

    def _get_optimizer(self) -> T.optim.Optimizer:
        return T.optim.AdamW(lr=self.args['train']['lr'], params=self.model.parameters(), betas=(0.9, 0.999), eps=1e-15)

    @abstractmethod
    def _get_scheduler(self) -> T.optim.lr_scheduler.LRScheduler:
        raise NotImplementedError()
    
    @abstractmethod
    def _get_loss_fn(self) -> T.nn.Module:
        raise NotImplementedError()

    @abstractmethod
    def step(self, *batch_data) -> T.Tensor:
        raise NotImplementedError()

    def write_summary(self, title: str, value: float, step: int):
        if self.can_log:
            self.summary_writer.add_scalar(title, value, step)
    
    def write_image(self, title: str, image: float, step: int):
        if self.can_log:
            self.summary_writer.add_image(title, image, step)

    def save_config(self):
        if not self.is_main_process:
            return
        
        config = self.args

        self.logger.info('======CONFIGURATIONS======')
        for k in config:
            self.logger.info(f'{k.upper()}')
            v: Dict = config[k]
            for ik, iv in v.items():
                self.logger.info(f'\t{ik.upper()}: {iv}')
        
        config_path = os.path.join(self.log_dir, 'config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        self.logger.info(f'Training config saved to {config_path}')

    def save_checkpoint(self, epoch: int, name: str = ''):
        if name != '':
            ckpt_id = name
        else:
            ckpt_id = f'epoch{epoch:02}_metric{self.tracker.last_metric:.4f}'
        self.model.save_checkpoint(self.ckpt_dir, ckpt_id)
    
    def load_checkpoint(self, ckpt_dir: str, ckpt_id: str):
        assert os.path.exists(ckpt_dir)
        self.model.load_checkpoint(ckpt_dir, ckpt_id)

    def train(self, dl: DataLoader, epoch: int):
        """
        Handles the training loop for a single epoch. 
        In this method, `yield` is to handle multiple validation phases within a single epoch.
        """
        self.logger.info('Training Phase')
        self.model.train()

        batch_losses = T.zeros(2, device=self.local_rank)
        pbar = tqdm(dl, disable=not self.is_main_process)

        for i, batch_data in enumerate(pbar):
            b_loss = self.step(*batch_data)
            self.model.backward(b_loss)
            self.model.step()

            for k in range(len(self.optim.param_groups)):
                self.write_summary(f'LR Scheduler/{k}', self.optim.param_groups[k]['lr'], self.tracker.step_counter)
            self.write_summary('Train/Batch Loss', b_loss, self.tracker.step_counter)

            self.tracker.inc_step_counter()
            yield i
            
            if not self.is_main_process:  # reset for gpu rank > 0
                batch_losses = T.zeros(2, device=self.local_rank)

            batch_losses[0] += b_loss
            batch_losses[1] += 1

            T.distributed.reduce(batch_losses, dst=0)
            avg_losses = batch_losses[0] / batch_losses[1]
            
            pbar.set_postfix({'Loss': f'{avg_losses:.4f}'})

        self.write_summary(f'Train/Loss', avg_losses, epoch)
        yield -1
    
    @T.no_grad()
    def validate(self, dl: DataLoader, epoch: int):
        """Handles the validation loop for a single epoch."""
        self.logger.info('Validation Phase')
        self.model.eval()

        batch_losses = T.zeros(2, device=self.local_rank)
        pbar = tqdm(dl, disable=not self.is_main_process)

        for batch_data in pbar:
            b_loss = self.step(*batch_data)
            self.tracker.inc_val_step_counter()
            
            if not self.is_main_process:  # reset for gpu rank > 0
                batch_losses = T.zeros(2, device=self.local_rank)

            batch_losses[0] += b_loss
            batch_losses[1] += 1

            T.distributed.reduce(batch_losses, dst=0)
            avg_losses = batch_losses[0] / batch_losses[1]
            
            pbar.set_postfix({'Loss': f'{avg_losses:.4f}'})

        T.distributed.broadcast(avg_losses, src=0)

        self.tracker.last_loss = avg_losses.item()
        self.tracker.last_metric = avg_losses.item()
        self.write_summary(f'Validation/Loss', avg_losses, epoch)
   
    def do_training(self, train_dataloader: DataLoader, val_dataloader: DataLoader):
        """Handles full training process for all epochs. Each epoch consists of training and validation phase."""
        self.logger.info('Begin Training')
        eval_per_epoch = self.args['train'].get('eval_per_epoch', 1)
        epoch = self.args['train'].get('epoch')
        patience = self.args['train'].get('patience', -1)
        ckpt_interval = self.args['train'].get('ckpt_interval', epoch)
        eval_idx = [len(train_dataloader) // eval_per_epoch * i for i in range(1, eval_per_epoch)]
        
        early_stop = False
        for epoch_idx in range(epoch):
            self.logger.info(f'Epoch {epoch_idx + 1}/{epoch}')
            for step in self.train(train_dataloader, epoch_idx):
                if step in eval_idx or step == -1:
                    self.validate(val_dataloader, epoch_idx)

                    self.save_checkpoint(epoch_idx + 1, 'last')
                    if self.tracker.is_metric_better(epoch_idx + 1):
                        self.save_checkpoint(epoch_idx + 1, 'best')
                    else:
                        if patience > 0 and epoch_idx + 1 - self.tracker.best_epoch > patience:
                            early_stop = True
                            break

            self.logger.info('Epoch Complete\n')

            if early_stop:
                self.logger.info(f'Early stopping. No improvement in validation metric for the last {patience} epochs.')
                break

        self.logger.info(f'Best result was seen in epoch {self.tracker.best_epoch} with metric value {self.tracker.best_metric:.4f}')
        
        if self.can_log:
            with open(os.path.join(self.log_dir, 'result.yaml'), 'w') as f:
                yaml.dump(self.tracker.to_dict(), f)
            self.logger.info(f'Result saved to {self.log_dir}/result.yaml')
            
