import argparse

import pandas as pd

from tqdm.auto import tqdm

import transformers
import torch
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb

from transformers import get_scheduler, get_cosine_schedule_with_warmup


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.texts = list((data['sentence_1'] + '[SEP]' + data['sentence_2']).values)
        if 'label' in data.columns:
            self.targets = data['label']
        else:
            self.targets = []

    def prepare_input(text):
        inputs = self.tokenizer.encode_plus(text, return_tensors=None, add_special_tokens=True, padding = 'max_length',truncation=True)
        for k, v in inputs.items():
            inputs[k] = torch.tensor(v, dtype=torch.long) # 텐서로 변환
        return inputs

    # 학습 및 추론 과정에서 데이터를 1개씩 꺼내오는 곳
    def __getitem__(self, idx):
        print(self.texts[idx])
        inputs = self.prepare_input(self.texts[idx])
        # 정답이 있다면 else문을, 없다면 if문을 수행합니다
        if len(self.targets) == 0:
            return inputs
        else:
            label = torch.tensor(self.targets[idx],dtype=torch.float)
            return inputs, label

    # 입력하는 개수만큼 데이터를 사용합니다
    def __len__(self):
        return len(self.texts)


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

    def tokenizing(self, dataframe):
        data = []
        for idx, item in tqdm(dataframe.iterrows(), desc='tokenizing_1', total=len(dataframe)):
            # 두 입력 문장을 [SEP] 토큰으로 이어붙여서 전처리합니다.
            text = '[SEP]'.join([item[text_column] for text_column in self.text_columns])
            outputs = self.tokenizer(text, add_special_tokens=True, padding='max_length', truncation=True)
            data.append(outputs)

        ###변경사항 2 <-> 1
        # for idx, item in tqdm(dataframe.iterrows(), desc='tokenizing_2', total=len(dataframe)):    
        #     text = '[SEP]'.join([item[text_column] for text_column in self.text_columns[::-1]])
        #     outputs = self.tokenizer(text, add_special_tokens=True, padding='max_length', truncation=True)
        #     data.append(outputs['input_ids'])

        return data



    def preprocessing(self, data):
        # 안쓰는 컬럼을 삭제합니다.
        data = data.drop(columns=self.delete_columns)

        # 타겟 데이터가 없으면 빈 배열을 리턴합니다.
        try:
            ##변경사항
            targets = data[self.target_columns].values.tolist() #+ data[self.target_columns].values.tolist()
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

            # # 학습데이터 준비
            # train_inputs, train_targets = self.preprocessing(train_data)

            # # 검증데이터 준비
            # val_inputs, val_targets = self.preprocessing(val_data)

            # train 데이터만 shuffle을 적용해줍니다, 필요하다면 val, test 데이터에도 shuffle을 적용할 수 있습니다
            self.train_dataset = Dataset(train_data)
            self.val_dataset = Dataset(val_data)
        else:
            # 평가데이터 준비   
            test_data = pd.read_csv(self.test_path)
            self.test_dataset = Dataset(test_data)

            predict_data = pd.read_csv(self.predict_path)
            self.predict_dataset = Dataset(predict_data)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, num_workers = 8,batch_size=self.batch_size, shuffle=args.shuffle)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, num_workers = 8,batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, num_workers = 8, batch_size=self.batch_size)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset, num_workers = 8, batch_size=self.batch_size)


class Model(pl.LightningModule):
    def __init__(self, model_name, lr):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.lr = lr

        # 사용할 모델을 호출합니다.
        # self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
        #     pretrained_model_name_or_path=model_name, num_labels=1)

        self.plm2 = transformers.AutoModel.from_pretrained(pretrained_model_name_or_path=model_name)
        self.fc = torch.nn.Linear(768,1)
        # Loss 계산을 위해 사용될 L1Loss를 호출합니다.
        self.loss_func = torch.nn.SmoothL1Loss(reduction='mean')

    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def feature(self,inputs):
        outputs = self.plm2(**inputs)
        last_hidden_states = outputs[0]
        feature = self.mean_pooling(outputs, inputs['attention_mask'])
        return feature
    
    def forward(self, x):
        feature = self.feature(x)
        outputs = self.fc(feature)
        return outputs

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
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay = 0.03)
        # scheduler = get_cosine_schedule_with_warmup(
        #         optimizer, num_warmup_steps=0, 
        #         num_training_steps= 9324 / args.max_epoch, num_cycles=0.5)
        return optimizer #, [scheduler]


if __name__ == '__main__':
    # 하이퍼 파라미터 등 각종 설정값을 입력받습니다
    # 터미널 실행 예시 : python3 run.py --batch_size=64 ...
    # 실행 시 '--batch_size=64' 같은 인자를 입력하지 않으면 default 값이 기본으로 실행됩니다
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='jhgan/ko-sroberta-multitask', type=str)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--max_epoch', default=10, type=int)
    parser.add_argument('--shuffle', default=True)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--train_path', default='../data/train.csv')
    parser.add_argument('--dev_path', default='../data/dev.csv')
    parser.add_argument('--test_path', default='../data/dev.csv')
    parser.add_argument('--predict_path', default='../data/test.csv')
    args = parser.parse_args(args=[])
    
    wandb.init(project="level1", entity="jjjjjun")
    wandb_logger = WandbLogger(project="level1")
    # dataloader와 model을 생성합니다.
    dataloader = Dataloader(args.model_name, args.batch_size, args.shuffle, args.train_path, args.dev_path,
                            args.test_path, args.predict_path)
    model = Model(args.model_name, args.learning_rate)
    
    ####수정
    checkpoint_callback = ModelCheckpoint(dirpath="/opt/ml/models/",save_top_k=2, monitor="val_loss")
    trainer = pl.Trainer(gpus=1, max_epochs=args.max_epoch, log_every_n_steps=1,precision=16, logger=wandb_logger, callbacks=[checkpoint_callback])

    # gpu가 없으면 'gpus=0'을, gpu가 여러개면 'gpus=4'처럼 사용하실 gpu의 개수를 입력해주세요
    # trainer = pl.Trainer(gpus=1, max_epochs=args.max_epoch, logger = wandb_logger,log_every_n_steps=1)

    # Train part
    trainer.fit(model=model, datamodule=dataloader)
    trainer.test(model=model, datamodule=dataloader)
    ###수정
    print(checkpoint_callback.best_model_path)


    # # 학습이 완료된 모델을 저장합니다.
    # torch.save(model, 'model.pt')
