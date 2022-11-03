import argparse

import pandas as pd
import numpy as np

from tqdm.auto import tqdm

import transformers
import torch
import torchmetrics
import pytorch_lightning as pl

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from sklearn.metrics.pairwise import cosine_similarity

import wandb
wandb.init(project="boostcamp_sts_competition")

from time import gmtime, strftime

import sys

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, model_name):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, max_length=160) 
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
    def __init__(self, model_name, batch_size, shuffle, train_path, dev_path, test_path, predict_path):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.shuffle = shuffle
        # self.k = k
        # self.split_seed = split_seed
        # self.n_folds = n_folds

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
            
            self.train_dataset = Dataset(train_data, self.model_name)
            self.val_dataset = Dataset(val_data, self.model_name)
            ###############################################################

        else:
            # 평가데이터 준비
            self.test_dataset = self.val_dataset   
            # test_data = pd.read_csv(self.test_path)
            # self.test_dataset = Dataset(test_data)

            predict_data = pd.read_csv(self.predict_path)
            self.predict_dataset = Dataset(predict_data, self.model_name)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=args.shuffle, num_workers=4)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset, batch_size=self.batch_size, num_workers=4)


class Model(pl.LightningModule):
    def __init__(self, model_name, lr, wd):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.lr = lr
        self.wd = wd

        # 사용할 모델을 호출합니다.
        # self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
        #     pretrained_model_name_or_path=model_name, num_labels=1)
        
        self.config = transformers.AutoConfig.from_pretrained(pretrained_model_name_or_path=model_name)
        self.plm = transformers.AutoModel.from_pretrained(pretrained_model_name_or_path=model_name, config=self.config)
        self.fc = torch.nn.Linear(1024,1)
        self.dropout = torch.nn.Dropout(p=0.3)

        # Loss 계산을 위해 사용될 L1Loss를 호출합니다.
        self.loss_func = torch.nn.MSELoss()

    def mean_pooling(self,model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings

    def feature(self,inputs):
        outputs = self.plm(**inputs)
        last_hidden_states = outputs[0]
        feature = self.mean_pooling(outputs, inputs['attention_mask'])
        return feature, last_hidden_states

    def forward(self, x):
        feature, last_hidden_states = self.feature(x)
        # print(last_hidden_states.size())
        # sys.exit()
        feature = self.dropout(feature)
        outputs = self.fc(feature)
        return outputs.view(-1), last_hidden_states.view(last_hidden_states.size()[0], last_hidden_states.size()[1]*last_hidden_states.size()[2])
        # view 함수 대신 linear layer 추가해서 차원 맞춰줘도 될듯..?
        

    def contrastive_loss(self, embedding, label, temp=0.3):
        """calculate the contrastive loss
        """
        embedding = embedding.cpu().detach().numpy()
        # print(embedding.size()) # 16, 512, 1024
        # cosine similarity between embeddings
        cosine_sim = cosine_similarity(embedding, embedding)
        # remove diagonal elements from matrix
        dis = cosine_sim[~np.eye(cosine_sim.shape[0], dtype=bool)].reshape(cosine_sim.shape[0], -1)
        # apply temprature to elements
        dis = dis / temp
        cosine_sim = cosine_sim / temp
        # apply exp to elements
        dis = np.exp(dis)
        cosine_sim = np.exp(cosine_sim)

        # calculate row sum
        row_sum = []
        for i in range(len(embedding)):
            row_sum.append(sum(dis[i]))
        # calculate outer sum
        contrastive_loss = 0
        for i in range(len(embedding)):
            n_i = label.tolist().count(label[i]) - 1
            inner_sum = 0
            # calculate inner sum
            for j in range(len(embedding)):
                if abs(label[i] - label[j])<=0.1 and i != j: # can fix
                    inner_sum = inner_sum + np.log(cosine_sim[i][j] / row_sum[i])
            if n_i != 0:
                contrastive_loss += (inner_sum / (-n_i))
            else:
                contrastive_loss += 0
        return contrastive_loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits,  hiden_state= self(x)
        mse_loss = self.loss_func(logits, y.float())
        con_loss = self.contrastive_loss(hiden_state, y.float())
        lam = 0.9 # can fix
        loss = (lam * con_loss) + (1 - lam) * (mse_loss)
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits,  hiden_state= self(x)
        mse_loss = self.loss_func(logits, y.float())
        con_loss = self.contrastive_loss(hiden_state, y.float())
        lam = 0.9 # can fix
        loss = (lam * con_loss) + (1 - lam) * (mse_loss)
        self.log("val_loss", loss)

        self.log("val_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits, hidden_state = self(x)

        self.log("test_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

    def predict_step(self, batch, batch_idx):
        x = batch
        logits, hidden_state = self(x)

        return logits.squeeze()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr,weight_decay=self.wd)
        return optimizer



if __name__ == '__main__':
    # 하이퍼 파라미터 등 각종 설정값을 입력받습니다
    # 터미널 실행 예시 : python3 run.py --batch_size=64 ...
    # 실행 시 '--batch_size=64' 같은 인자를 입력하지 않으면 default 값이 기본으로 실행됩니다
    parser = argparse.ArgumentParser()
    # klue/roberta-small
    # klue/roberta-base
    # klue/bert-base
    parser.add_argument('--model_name', default='klue/roberta-large', type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--max_epoch', default=10, type=int)
    parser.add_argument('--shuffle', default=True)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--weight_decay',default=1e-4, type=float)
    # parser.add_argument('--train_path', default='../data/train.csv')
    # parser.add_argument('--dev_path', default='../data/dev.csv')
    parser.add_argument('--data', default='all')
    parser.add_argument('--test_path', default='../data/dev.csv')
    parser.add_argument('--predict_path', default='../data/test.csv')
    args = parser.parse_args(args=[])

    print(args.data)

    if args.data == 'nsmc':
        train_path = '../data/nsmc_train.csv'
        dev_path = '../data/nsmc_dev.csv'
    elif args.data == 'petition':
        print('here')
        train_path = '../data/petition_train.csv'
        dev_path = '../data/petition_dev.csv'
    elif args.data == 'slack':
        train_path = '../data/slack_train.csv'
        dev_path = '../data/slack_dev.csv'
    elif args.data == 'all':
        train_path = '../data/train.csv'
        dev_path = '../data/dev.csv'
    elif args.data == 'all_new':
        train_path = '../data/new_train.csv'
        dev_path = '../data/dev.csv'

    


    # dataloader와 model을 생성합니다.
    dataloader = Dataloader(args.model_name, args.batch_size, args.shuffle, train_path, dev_path,
                            args.test_path, args.predict_path)
    model = Model(args.model_name, args.learning_rate, args.weight_decay)

    # wandb logger 설정
    wandb_logger = WandbLogger(name=f'{args.model_name}+{args.batch_size}+{args.learning_rate}', project="boostcamp_sts_competition")

    checkpoint_callback = ModelCheckpoint(dirpath=f"/opt/ml/models/roberta-large-contrastive/{args.data}", save_top_k=2, monitor="val_loss")
    # gpu가 없으면 'gpus=0'을, gpu가 여러개면 'gpus=4'처럼 사용하실 gpu의 개수를 입력해주세요
    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=args.max_epoch, log_every_n_steps=1, logger=wandb_logger, precision=16, callbacks=[checkpoint_callback])

    # Train part
    trainer.fit(model=model, datamodule=dataloader)
    trainer.test(model=model, datamodule=dataloader)

    print(checkpoint_callback.best_model_path)

    # num_folds = 5
    # split_seed = 22

    # wandb.init(project="level1", entity="jjjjjun")
    # wandb_logger = WandbLogger(project="level1")

    
    # results = []

    # for k in range(num_folds):
    #     print(f'=======================fold {k}==========================')
    #     dataloader = Dataloader(args.model_name, args.batch_size, args.shuffle, args.train_path, args.dev_path,
    #                             args.test_path, args.predict_path, k=k ,n_folds=num_folds,split_seed=split_seed)
    #     dataloader.prepare_data()
    #     dataloader.setup()
    #     model = Model(args.model_name, args.learning_rate)
    #     checkpoint_callback = ModelCheckpoint(dirpath="/opt/ml/models/",save_top_k=2, monitor="val_loss",filename=f"kfold-{k:02d}")
    #     trainer = pl.Trainer(gpus=1, max_epochs= args.max_epoch, precision=16,log_every_n_steps=1, logger=wandb_logger, callbacks=[checkpoint_callback])
    #     trainer.fit(model = model, datamodule = dataloader)
    #     score = trainer.test(model=model, datamodule=dataloader)
    #     results.extend(score)
    #     print(checkpoint_callback.best_model_path)

    #     predictions = trainer.predict(model=model, datamodule=dataloader)
    #     predictions = list(float(i) for i in torch.cat(predictions))

    #     output = pd.read_csv('../data/sample_submission.csv')
    #     output['target'] = predictions
    #     output.to_csv(f'output-{k} fold.csv', index=False)

    # print(results)
    # result = [x['test_pearson'] for x in results]
    # score = sum(result) / num_folds
    # print("K fold Test pearson",score)

    # wandb.finish()

    # 학습이 완료된 모델을 저장합니다.
    # save_name = re.sub('/', '_', args.model_name)
    torch.save(model, f'/opt/ml/models/roberta-large-meanpooling/{args.data}/{args.data}_{strftime("%m-%d-%H-%M", gmtime())}.pt')

    print('<< BEST CHECKPOINT PATH >>')
    print(checkpoint_callback.best_model_path)
    print()