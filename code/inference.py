
import argparse
from tqdm.auto import tqdm

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping

import wandb

from datamodule import Dataloader
from model import *
import pandas as pd


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
    parser.add_argument('--augmentation', default=False, type=bool)
    parser.add_argument('--num_aug', default=2, type=int)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--train_path', default='/opt/ml/data/train.csv')
    parser.add_argument('--dev_path', default='/opt/ml/data/dev.csv')
    parser.add_argument('--test_path', default='/opt/ml/data/dev.csv')
    parser.add_argument('--predict_path', default='/opt/ml/data/test.csv')
    parser.add_argument('--ensamble', default=True)
    
    args = parser.parse_args(args=[])

    
    
    # dataloader와 model을 생성합니다.
    dataloader = Dataloader(
        args.model_name, 
        args.batch_size, 
        args.shuffle, 
        args.train_path, 
        args.dev_path,
        args.test_path, 
        args.predict_path,
        args.augmentation,
        args.num_aug
    )

    trainer = pl.Trainer(gpus=1, max_epochs=args.max_epoch, log_every_n_steps=1)

    # checkpoint load
    if args.ensamble:
        model = EnsambleModel(
        regression_bert_base_model_path='./models/regression-bert-base-model-epoch-end.ckpt',
        regression_roberta_base_model_path='./models/regression-roberta-base-model-epoch-end.ckpt'
    )    
    else:
        model = RegressionBertBaseModel.load_from_checkpoint('./models/regression-bert-base-model-epoch-end.ckpt')
        
    predictions = trainer.predict(model=model, datamodule=dataloader)

    predictions = list(round(float(i), 1) for i in torch.cat(predictions))
    
    output = pd.read_csv('/opt/ml/data/sample_submission.csv')
    output['target'] = predictions
    output.loc[output['target'] < 0, 'target'] = float(int(0))
    output.loc[output['target'] > 5, 'target'] = float(5)
    
    output.to_csv('output.csv', index=False)
