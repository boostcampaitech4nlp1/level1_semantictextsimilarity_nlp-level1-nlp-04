import argparse

import pandas as pd

from tqdm.auto import tqdm

import transformers
import torch
import torchmetrics
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.model_selection import StratifiedKFold , KFold

import wandb

from transformers import get_scheduler, get_cosine_schedule_with_warmup


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained('klue/roberta-small', max_length=160) 
        self.texts = (data['sentence_1'] + '[SEP]' + data['sentence_2']).values
        if 'label' in data.columns:
            self.targets = data['label'].values
        else:
            self.targets = []

    def prepare_input(self,text):
        inputs = self.tokenizer.encode_plus(text, return_tensors=None,
        add_special_tokens=True, padding = 'max_length',truncation=True)
        for k, v in inputs.items():
            inputs[k] = torch.tensor(v, dtype=torch.long) # 텐서로 변환
        return inputs

    # 학습 및 추론 과정에서 데이터를 1개씩 꺼내오는 곳
    def __getitem__(self, idx):
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
    def __init__(self, model_name, batch_size, shuffle, train_path, dev_path, test_path, predict_path,k,n_folds,split_seed):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.k = k
        self.split_seed = split_seed
        self.n_folds = n_folds

        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.predict_path = predict_path

        self.train_dataset = None
        self.val_dataset = None
        self.total_datset = None
        self.test_dataset = None
        self.predict_dataset = None

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, max_length=160)
        self.target_columns = ['label']
        self.delete_columns = ['id']
        self.text_columns = ['sentence_1', 'sentence_2']

    def setup(self, stage='fit'):
        if stage == 'fit':
            # 학습 데이터와 검증 데이터셋을 호출합니다
            train_data = pd.read_csv(self.train_path)
            val_data = pd.read_csv(self.dev_path)

            #토탈 데이터셋
            total_data = pd.concat([train_data,val_data],axis = 0)
            total_data['simple_label'] = total_data['label'].apply(lambda x : int(x))
            total_data = total_data.reset_index().drop(columns='index')
            ############################KFold##############################
            skf = StratifiedKFold(n_splits=self.n_folds, shuffle = self.shuffle, random_state=self.split_seed)
  
            for n_fold, (_,v_idx) in enumerate(skf.split(total_data,total_data['simple_label'])):
                total_data.loc[v_idx,'fold'] = n_fold

            train_data = total_data[total_data['fold'] != self.k]
            val_data = total_data[total_data['fold'] == self.k]
            
            self.train_dataset = Dataset(train_data)
            self.val_dataset = Dataset(val_data)
            ###############################################################

        else:
            # 평가데이터 준비
            self.test_dataset = self.val_dataset   
            # test_data = pd.read_csv(self.test_path)
            # self.test_dataset = Dataset(test_data)

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
        self.config = transformers.AutoConfig.from_pretrained(pretrained_model_name_or_path=model_name)
        self.plm2 = transformers.AutoModel.from_pretrained(pretrained_model_name_or_path=model_name)
        self.fc = torch.nn.Linear(768,1)
        self.dropout = torch.nn.Dropout(p=0.3)

        # Loss 계산을 위해 사용될 L1Loss를 호출합니다.
        self.loss_func = torch.nn.SmoothL1Loss(reduction='mean')

    def mean_pooling(self,model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings

    def feature(self,inputs):
        outputs = self.plm2(**inputs)
        last_hidden_states = outputs[0]
        feature = self.mean_pooling(outputs, inputs['attention_mask'])
        return feature
    
    def forward(self, x):
        feature = self.feature(x)
        feature = self.dropout(feature)
        outputs = self.fc(feature)
        return outputs.view(-1)

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
        return torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze())

    def predict_step(self, batch, batch_idx):
        x = batch
        logits = self(x)

        return logits.squeeze()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay = 0.01)
        return optimizer


if __name__ == '__main__':
    # 하이퍼 파라미터 등 각종 설정값을 입력받습니다
    # 터미널 실행 예시 : python3 run.py --batch_size=64 ...
    # 실행 시 '--batch_size=64' 같은 인자를 입력하지 않으면 default 값이 기본으로 실행됩니다
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='klue/roberta-base', type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--max_epoch', default=8, type=int)
    parser.add_argument('--shuffle', default=True)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--train_path', default='../data/train.csv')
    parser.add_argument('--dev_path', default='../data/dev.csv')
    parser.add_argument('--test_path', default='../data/dev.csv')
    parser.add_argument('--predict_path', default='../data/test.csv')
    args = parser.parse_args(args=[])
    
    num_folds = 10
    split_seed = 22

    wandb.init(project="level1", entity="jjjjjun")
    wandb_logger = WandbLogger(project="level1")

    
    results = []

    for k in range(num_folds):
        print(f'=======================fold {k}==========================')
        dataloader = Dataloader(args.model_name, args.batch_size, args.shuffle, args.train_path, args.dev_path,
                                args.test_path, args.predict_path, k=k ,n_folds=num_folds,split_seed=split_seed)
        dataloader.prepare_data()
        dataloader.setup()
        model = Model(args.model_name, args.learning_rate)
        checkpoint_callback = ModelCheckpoint(dirpath="/opt/ml/models/",save_top_k=2, monitor="val_loss",filename=f"kfold-{k:02d}")
        trainer = pl.Trainer(gpus=1, max_epochs= args.max_epoch, precision=16,log_every_n_steps=1, logger=wandb_logger, callbacks=[checkpoint_callback])
        trainer.fit(model = model, datamodule = dataloader)
        score = trainer.test(model=model, datamodule=dataloader)
        results.extend(score)
        print(checkpoint_callback.best_model_path)
    
    print(results)
    result = [x['test_pearson'] for x in results]
    score = sum(result) / num_folds
    print("K fold Test pearson",score)

    wandb.finish()