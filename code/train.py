import argparse

import pandas as pd

from tqdm.auto import tqdm

import transformers
import torch
import torchmetrics
import pytorch_lightning as pl

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer

# [added] for saving best model
from pytorch_lightning.callbacks import ModelCheckpoint

import re

# [added] to ignore warning
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# [added]
import wandb

# [added]
# 성능 안좋아짐.. 삭제

# from pykospacing import Spacing

# [added]

from time import gmtime, strftime






class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets=[]):
        self.inputs = inputs
        self.targets = targets

    # 학습 및 추론 과정에서 데이터를 1개씩 꺼내오는 곳
    def __getitem__(self, idx):
        # 정답이 있다면 else문을, 없다면 if문을 수행합니다
        if len(self.targets) == 0:
            return torch.tensor(self.inputs[idx])
        else:
            return torch.tensor(self.inputs[idx]), torch.tensor(self.targets[idx])

    # 입력하는 개수만큼 데이터를 사용합니다
    def __len__(self):
        return len(self.inputs)


class Dataloader(pl.LightningDataModule):
    def __init__(self, model_name, batch_size, shuffle, train_path, dev_path, test_path, predict_path):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.predict_path = predict_path

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, max_length=160)
        self.target_columns = ['label']
        self.delete_columns = ['id']
        self.text_columns = ['sentence_1', 'sentence_2']

    # def tokenizing(self, dataframe):
    #     data = []
    #     for idx, item in tqdm(dataframe.iterrows(), desc='tokenizing', total=len(dataframe)):
    #         # 두 입력 문장을 [SEP] 토큰으로 이어붙여서 전처리합니다.
    #         text = '[SEP]'.join([item[text_column] for text_column in self.text_columns])
    #         # [added] - 전처리 
    #         text = re.sub('ㅋ+|ㅎ+', '웃음', text)
    #         text = re.sub('ㅜ+|ㅠ+', '슬픔', text)
    #         text = re.sub(';+', '당황', text)
            
    #         outputs = self.tokenizer(text, add_special_tokens=True, padding='max_length', truncation=True)
    #         data.append(outputs['input_ids'])
    #     return data

    # def preprocessing(self, data):
    #     # 안쓰는 컬럼을 삭제합니다.
    #     data = data.drop(columns=self.delete_columns)

    #     # [added] : 띄어쓰기 전처리
    #     spacing = Spacing()
    #     # kospacing_sent = spacing(new_sent) 

    #     for idx, item in data.iterrows():
    #         item['sentence_1'] = spacing(item['sentence_1'])
    #         item['sentence_2'] = spacing(item['sentence_2'])

    #     # 타겟 데이터가 없으면 빈 배열을 리턴합니다.
    #     try:
    #         targets = data[self.target_columns].values.tolist()
    #     except:
    #         targets = []
    #     # 텍스트 데이터를 전처리합니다.
    #     inputs = self.tokenizing(data)

    #     return inputs, targets

    def tokenizing(self, dataframe):
        data = []
        for idx, item in tqdm(dataframe.iterrows(), desc='tokenizing', total=len(dataframe)):
            # 두 입력 문장을 [SEP] 토큰으로 이어붙여서 전처리합니다.
            text = '[SEP]'.join([item[text_column] for text_column in self.text_columns])
            outputs = self.tokenizer(text, add_special_tokens=True, padding='max_length', truncation=True)
            data.append(outputs['input_ids'])
        return data

    def preprocessing(self, data):
        # 안쓰는 컬럼을 삭제합니다.
        data = data.drop(columns=self.delete_columns)

        # 타겟 데이터가 없으면 빈 배열을 리턴합니다.
        try:
            targets = data[self.target_columns].values.tolist()
        except:
            targets = []
        # 텍스트 데이터를 전처리합니다.
        inputs = self.tokenizing(data)

        return inputs, targets

    def setup(self, stage='fit'):
        if stage == 'fit':
            # 학습 데이터와 검증 데이터셋을 호출합니다
            train_data = pd.read_csv(self.train_path)
            val_data = pd.read_csv(self.dev_path)

            # 학습데이터 준비
            train_inputs, train_targets = self.preprocessing(train_data)

            # 검증데이터 준비
            val_inputs, val_targets = self.preprocessing(val_data)

            # train 데이터만 shuffle을 적용해줍니다, 필요하다면 val, test 데이터에도 shuffle을 적용할 수 있습니다
            self.train_dataset = Dataset(train_inputs, train_targets)
            self.val_dataset = Dataset(val_inputs, val_targets)
        else:
            # 평가데이터 준비
            test_data = pd.read_csv(self.test_path)
            test_inputs, test_targets = self.preprocessing(test_data)
            self.test_dataset = Dataset(test_inputs, test_targets)

            predict_data = pd.read_csv(self.predict_path)
            predict_inputs, predict_targets = self.preprocessing(predict_data)
            self.predict_dataset = Dataset(predict_inputs, [])

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=args.shuffle, num_workers=8)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=8)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=8)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset, batch_size=self.batch_size, num_workers=8)


class Model(pl.LightningModule):
    def __init__(self, model_name, lr, wd):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.lr = lr
        self.wd = wd

        # 사용할 모델을 호출합니다.
        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_name, num_labels=1)
        # Loss 계산을 위해 사용될 L1Loss를 호출합니다.
        # [edited] -> MSE loss로 변경
        self.loss_func = torch.nn.MSELoss()

    def forward(self, x):
        x = self.plm(x)['logits']

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        self.log("val_loss", loss)

        self.log("val_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        self.log("test_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

    def predict_step(self, batch, batch_idx):
        x = batch
        logits = self(x)

        return logits.squeeze()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        return optimizer


if __name__ == '__main__':
    # 하이퍼 파라미터 등 각종 설정값을 입력받습니다
    # 터미널 실행 예시 : python3 run.py --batch_size=64 ...
    # 실행 시 '--batch_size=64' 같은 인자를 입력하지 않으면 default 값이 기본으로 실행됩니다
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='jhgan/ko-sroberta-multitask', type=str)
    # bespin-global/klue-sroberta-base-continue-learning-by-mnr
    # klue/roberta-large -> memory error 발생
    # klue/roberta-small
    # klue/roberta-base
    # klue/bert-base
    # eliza-dukim/bert-base-finetuned-sts
    # jhgan/ko-sroberta-multitask
    # snunlp/KR-SBERT-V40K-klueNLI-augSTS
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--max_epoch', default=10, type=int)
    parser.add_argument('--shuffle', default=True)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--weight_decay',default=1e-5, type=float)
    parser.add_argument('--train_path', default='../data/train.csv')
    parser.add_argument('--dev_path', default='../data/dev.csv')
    parser.add_argument('--test_path', default='../data/dev.csv')
    parser.add_argument('--predict_path', default='../data/dev.csv')
    parser.add_argument('--sweep', action='store_true')
    args = parser.parse_args(args=[])

    # [added]
    # wandb_logger = WandbLogger(project="boostcamp_level_1")
    wandb_logger = WandbLogger(name=f'{args.model_name}+{args.batch_size}+{args.learning_rate}+{args.weight_decay}+16',project="boostcamp_level_1")


    # dataloader와 model을 생성합니다.
    dataloader = Dataloader(args.model_name, args.batch_size, args.shuffle, args.train_path, args.dev_path,
                            args.test_path, args.predict_path)
    model = Model(args.model_name, args.learning_rate, args.weight_decay)

    # gpu가 없으면 'gpus=0'을, gpu가 여러개면 'gpus=4'처럼 사용하실 gpu의 개수를 입력해주세요
    
    # [added] : for saving top 2 best models
    checkpoint_callback = ModelCheckpoint(dirpath="/opt/ml/models/", save_top_k=2, monitor="val_loss")

    # [added] : sweeping
    if args.sweep:

        def sweep_train(config=None):
            trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=args.max_epoch, logger=wandb_logger, log_every_n_steps=1)
            trainer.fit(model=model, datamodule=dataloader)
            trainer.test(model=model, datamodule=dataloader)


        sweep_config = {
            'method': 'random', # random: 임의의 값의 parameter 세트를 선택
            'parameters': {
                'lr':{
                    'distribution': 'uniform',  # parameter를 설정하는 기준을 선택합니다. uniform은 연속적으로 균등한 값들을 선택합니다.
                    'min':1e-5,                 # 최소값을 설정합니다.
                    'max':1e-4                  # 최대값을 설정합니다.
                },
                'batch_size':{
                    'distribution':'uniform',
                    'min':8,
                    'max':32
                },
            'metric' : {'name':'val_pearson', 'goal':'maximize'} # pearson 점수가 최대화가 되는 방향으로 학습을 진행합니다.
            }
        }
        sweep_id = wandb.sweep(
            sweep=sweep_config,     # config 딕셔너리를 추가합니다.
            project='boostcamp_level_1'  # project의 이름을 추가합니다.
        )
        wandb.agent(
            sweep_id=sweep_id,      # sweep의 정보를 입력하고
            function=sweep_train,   # train이라는 모델을 학습하는 코드를
            count=5                 # 총 5회 실행해봅니다.
        )
        
    else:
    
        # [edited] - add logger=wandb_logger / precision = 16
        trainer = pl.Trainer(gpus=1, max_epochs=args.max_epoch, log_every_n_steps=1, logger=wandb_logger, \
                             precision=16, callbacks=[checkpoint_callback])


        # Train part
        trainer.fit(model=model, datamodule=dataloader)
        trainer.test(model=model, datamodule=dataloader)


    # 학습이 완료된 모델을 저장합니다.
    save_name = re.sub('/', '_', args.model_name)
    torch.save(model, f'/opt/ml/models/{save_name}_{strftime("%m-%d-%H-%M", gmtime())}_bf16.pt')

    print('<< BEST CHECKPOINT PATH >>')
    print(checkpoint_callback.best_model_path)

    # [added] : 예측값과 정답 차이 확인하기

    model = Model.load_from_checkpoint(checkpoint_path=checkpoint_callback.best_model_path).eval()

    predictions = trainer.predict(model=model, datamodule=dataloader) # dev 데이터셋으로 predict
    preds = torch.cat(predictions)
    wrongs = []

    for i, pred in enumerate(preds):
        # test dataset에서 i번째에 해당하는 input값과 target값을 가져옵니다
        input_ids, target = dataloader.test_dataset.__getitem__(i)
        # 예측값과 정답값이 다를 경우 기록합니다.
        if round(pred.item()) - round(target.item()) >= 0.5:
            wrongs.append([dataloader.tokenizer.decode(input_ids).replace(' [PAD]', ''), pred.item(), target.item()])

    wrong_df = pd.DataFrame(wrongs, columns=['text', 'pred', 'target'])

    wrong_df.to_csv(f'/opt/ml/code/wrongs/{re.sub("/", "-", args.model_name)}_wrong.csv')
