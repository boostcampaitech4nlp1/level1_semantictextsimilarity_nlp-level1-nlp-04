import argparse

import pandas as pd

from tqdm.auto import tqdm

import transformers
import torch
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

import wandb
from model import RegressionModel
from datamodule import Dataloader




if __name__ == '__main__':
    # 하이퍼 파라미터 등 각종 설정값을 입력받습니다
    # 터미널 실행 예시 : python3 run.py --batch_size=64 ...
    # 실행 시 '--batch_size=64' 같은 인자를 입력하지 않으면 default 값이 기본으로 실행됩니다
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='klue/roberta-base', type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--max_epoch', default=10, type=int)
    parser.add_argument('--shuffle', default=True)
    parser.add_argument('--norm', default=1, type=int)
    parser.add_argument('--num_aug', default=2, type=int)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--cls_weight', default=1e-3, type=float)
    parser.add_argument('--train_path', default='/opt/ml/data/train.csv')
    parser.add_argument('--dev_path', default='/opt/ml/data/dev.csv')
    parser.add_argument('--test_path', default='/opt/ml/data/dev.csv')
    parser.add_argument('--predict_path', default='/opt/ml/data/test.csv')
    args = parser.parse_args(args=[])

    try:
        wandb.login(key='4c0a01eaa2bd589d64c5297c5bc806182d126350')
    except:
        anony = "must"
        print('If you want to use your W&B account, go to Add-ons -> Secrets and provide your W&B access token. Use the Label name as wandb_api. \nGet your W&B access token from here: https://wandb.ai/authorize')
    
    # wandb sweep config
    sweep_config = {
        'method': 'random',
        'parameters': {
            'lr': {
                'distribution': 'uniform',
                'min': 1e-5,
                'max': 1e-4
            },
            'norm':{
                'values': [1, 2]
            },
            'cls_weight':{
                'distribution': 'uniform',
                'min': 1e-5,
                'max': 1e-3
            }
        },
        'metric': {
            'name': 'val_pearson',
            'goal': 'maximize'
        }
    }

    # create sweep
    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project='project',
    )
    
    def sweep_train(config=None):
        wandb.init(project="project", name=f"{args.model_name}", config=config)
        wandb_logger = WandbLogger('project')
        config = wandb.config
        
        dataloader = Dataloader(
            args.model_name, 
            args.batch_size, 
            args.shuffle, 
            args.train_path, 
            args.dev_path, 
            args.test_path, 
            args.predict_path,
            args.num_aug
        )
        model = RegressionModel(args.model_name, args.learning_rate, args.norm, args.cls_weight)
        earlystop_callback = EarlyStopping(
            monitor='val_total_loss',
            mode='min'
        )
        
        trainer = pl.Trainer(
            accelerator='gpu',
            devices=1,
            max_epochs=args.max_epoch, 
            logger=wandb_logger,
            log_every_n_steps=1, 
            callbacks=[earlystop_callback]
        )

        # Train part
        trainer.fit(model=model, datamodule=dataloader)
        
        # Test part
        trainer.test(model=model, datamodule=dataloader)
    
    wandb.agent(
        sweep_id=sweep_id,
        function=sweep_train,
        count=10
    )
    
    wandb.finish()