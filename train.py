import os
import argparse
import yaml
import random
import torch as T
import numpy as np
import deepspeed
from typing import Dict
from utils.logger import *
from torch.utils.data import DataLoader
from data.example_data import ExampleData
from model.example_model import ExampleModel
from trainer.example_trainer import ExampleTrainer

def main(train_args: Dict):
    setup_logging()
    seed_everything(train_args['train']['seed'])
    deepspeed.init_distributed()
    
    logger = get_logger(__name__, int(os.environ['RANK']))

    logger.info('Instantiating model and trainer agent')
    model = ExampleModel(**train_args['model'])
    trainer = ExampleTrainer(
        model, 
        int(os.environ['LOCAL_RANK']), 
        int(os.environ['RANK']),
        train_args, 
        log_enabled=not train_args['train']['no_save']
    )

    logger.info('Preparing dataset')
    train_dataset = ExampleData(**train_args['data'], validation=False)
    val_dataset = ExampleData(**train_args['data'], validation=True)
    logger.info(f'Train dataset size: {len(train_dataset)}')
    logger.info(f'Val dataset size: {len(val_dataset)}')

    world_size = int(os.environ['WORLD_SIZE'])
    grad_acc_steps = train_args['train']['deepspeed']['gradient_accumulation_steps']
    micro_batch_size = train_args['train']['deepspeed']['train_micro_batch_size_per_gpu']
    logger.info(f'Using {world_size} GPU(s)')
    logger.info(f'Gradient accumulation steps: {grad_acc_steps}')
    logger.info(f'Batch size each GPU: {micro_batch_size}')
    logger.info(f'Actual batch size: {world_size * micro_batch_size * grad_acc_steps}')
    
    if train_args['train'].get('model_path') is not None:
        trainer.load_checkpoint(train_args['model_path'])
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=micro_batch_size,
        shuffle=True,
        num_workers=train_args['train']['n_workers'],
        pin_memory=True,
        persistent_workers=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=micro_batch_size,
        shuffle=False,
        num_workers=train_args['train']['n_workers'],
        pin_memory=True,
        persistent_workers=True,
    )

    trainer.do_training(train_dataloader, val_dataloader)

def seed_everything(seed: int):    
    random.seed(seed)
    np.random.seed(seed)
    T.manual_seed(seed)
    T.cuda.manual_seed(seed)
    T.backends.cudnn.deterministic = True
    T.backends.cudnn.benchmark = True

def get_args_parser():
    parser = argparse.ArgumentParser('QuickTorch Train', add_help=False)
    parser.add_argument('--uid', type=str, help='unique id for the run', default=None)
    parser.add_argument('--config', type=str, help='path to json config', default='config/example_config.yaml')
    parser.add_argument('--model-path', type=str, help='ckpt path to continue', default=None)
    parser.add_argument('--patience', type=int, help='patience for early stopping', default=-1)
    parser.add_argument('--seed', type=int, help='random seed', default=None)
    parser.add_argument('--no-ddp', action='store_true', help='disable DDP')
    parser.add_argument('--no-save', action='store_true', help='disable logging and checkpoint saving (for debugging)')
    parser.add_argument('--local_rank', type=int, help='local rank (automatically set)')

    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args_parser()
    train_args = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    
    if args.seed is not None:
        train_args['train']['seed'] = args.seed
    elif train_args['train'].get('seed', None) is None:
        train_args['train']['seed'] = random.randint(0, 1000000)
    if args.uid is not None:
        train_args['train']['uid'] = args.uid
    if args.model_path is not None:
        train_args['train']['model_path'] = args.model_path
    if args.patience != -1:
        train_args['train']['patience'] = args.patience 
    train_args['train']['no_ddp'] = args.no_ddp
    train_args['train']['no_save'] = args.no_save
    
    main(train_args)
