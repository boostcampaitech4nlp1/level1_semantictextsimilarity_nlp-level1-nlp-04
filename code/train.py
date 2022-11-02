
import argparse
from tqdm.auto import tqdm

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping

import wandb

from datamodule import Dataloader
from model import ClassificationModel, RegressionBertBaseModel, RegressionRobertaBaseModel, RegulationModel, RegressionModel

import gc


if __name__ == '__main__':
    # 하이퍼 파라미터 등 각종 설정값을 입력받습니다
    # 터미널 실행 예시 : python3 run.py --batch_size=64 ...
    # 실행 시 '--batch_size=64' 같은 인자를 입력하지 않으면 default 값이 기본으로 실행됩니다
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', default='fit', type=str)         # fit / test / predict
    parser.add_argument('--model_name', default='klue/roberta-base', type=str)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--max_epoch', default=10, type=int)
    parser.add_argument('--shuffle', default=True)
    parser.add_argument('--norm', default=1, type=int)
    parser.add_argument('--augmentation', default=False, type=bool)  # 데이터 증강 하기 싫으면 False
    parser.add_argument('--num_aug', default=2, type=int)           # 한 문장 당 데이터 증강 개수
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--train_path', default='/opt/ml/data/train.csv')
    parser.add_argument('--dev_path', default='/opt/ml/data/dev.csv')
    parser.add_argument('--test_path', default='/opt/ml/data/dev.csv')
    parser.add_argument('--predict_path', default='/opt/ml/data/test.csv')
    args = parser.parse_args(args=[])

    # try:
    #     wandb.login(key='4c0a01eaa2bd589d64c5297c5bc806182d126350')
    # except:
    #     anony = "must"
    #     print('If you want to use your W&B account, go to Add-ons -> Secrets and provide your W&B access token. Use the Label name as wandb_api. \nGet your W&B access token from here: https://wandb.ai/authorize')

    # wandb.init(project="project", name= f"{args.model_name}")
    # wandb_logger = WandbLogger('project')
    
    
    # dataloader
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
    
    checkpoint_callback = ModelCheckpoint(
        dirpath='./models',
        filename='model+{epoch}+{val_loss:.2f}',
        monitor='val_total_loss',
        save_top_k=2
    )
    earlystopping_callback = EarlyStopping(
        monitor='val_distance_loss',
        mode='min'
    )
    
    # Trainer
    # regression_model = RegressionModel(args.model_name, args.learning_rate, args.norm)
    # regression_bert_base_model = RegressionBertBaseModel(args.learning_rate, args.norm)
    regression_roberta_base_model = RegressionRobertaBaseModel(args.learning_rate, args.norm)

    regression_bert_base_trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        max_epochs=args.max_epoch,
        # logger=wandb_logger,
        log_every_n_steps=1,
        callbacks=[
            # checkpoint_callback,
            earlystopping_callback
        ]
    )
    
    regression_roberta_base_trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        max_epochs=args.max_epoch,
        # logger=wandb_logger,
        log_every_n_steps=1,
        callbacks=[
            # checkpoint_callback,
            earlystopping_callback
        ]
    )
    
    # regression_trainer / train + validation
    # regression_bert_base_trainer.fit(model=regression_bert_base_model, datamodule=dataloader)
    # regression_bert_base_trainer.save_checkpoint('./models/regression-bert-base-model-epoch-end.ckpt')
    
    # # test
    # regression_bert_base_trainer.test(model=regression_bert_base_model, datamodule=dataloader)
    
    
    gc.collect()
    torch.cuda.empty_cache()
    
    # regression_trainer / train + validation
    regression_roberta_base_trainer.fit(model=regression_roberta_base_model, datamodule=dataloader)
    regression_roberta_base_trainer.save_checkpoint('./models/regression-roberta-base-model-epoch-end.ckpt')

    # test
    regression_roberta_base_trainer.test(model=regression_roberta_base_model, datamodule=dataloader)    
    
    
    
    # classification_model = ClassificationModel(args.model_name, args.learning_rate)
    # classification_trainer = pl.Trainer(
    #     accelerator='gpu',
    #     devices=1,
    #     max_epochs=args.max_epoch,
    #     # logger=wandb_logger,
    #     log_every_n_steps=1,
    # )
    
    # # classification_trainer / train + validation
    # classification_trainer.fit(model=classification_model, datamodule=dataloader)
    # classification_trainer.save_checkpoint('./models/classification-model-epoch-end.ckpt')
    
    # # test
    # classification_trainer.test(model=classification_model, datamodule=dataloader)
    
    # regulation_model
    # regulation_model = RegulationModel(
    #     model_name=args.model_name,
    #     distance_model_path='./models/regression-model-epoch-end.ckpt',
    #     cls_model_path='./models/classification-model-epoch-end.ckpt',
    #     lr=args.learning_rate,
    #     norm=args.norm
    # )
    
    # regulation_trainer = pl.Trainer(
    #     accelerator='gpu',
    #     devices=1,
    #     max_epochs=10,
    #     logger=wandb_logger,
    #     log_every_n_steps=1
    # )
    
    # regulation_trainer.fit(model=regulation_model, datamodule=dataloader)
    # regulation_trainer.save_checkpoint('./models/ensamble-model-epoch-end.ckpt')
    
    # regulation_trainer.test(model=regulation_model, datamodule=dataloader)
    
    
    # 학습이 완료된 모델을 저장합니다.
    # torch.save(regression_model, 'model-epoch-end.pt')