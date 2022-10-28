import re
import pytorch_lightning as pl
import torch
import transformers
from tqdm.auto import tqdm
import pandas as pd
from pykospacing import Spacing


import util
import augmentation

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
        
        # main data
        self.target_columns = ['label']
        self.delete_columns = ['id']
        self.text_columns = ['sentence_1', 'sentence_2']
        
        # pyspacing
        self.spacing = Spacing()
        

    def tokenizing(self, dataframe):
        data = []
        for idx, item in tqdm(dataframe.iterrows(), desc='tokenizing', total=len(dataframe)):
            # 두 입력 문장을 [SEP] 토큰으로 이어붙여서 전처리합니다.
            
            # text = '[SEP]'.join(
            #     [util.get_only_korean(item[text_column]) for text_column in self.text_columns]
            #     # [item[text_column] for text_column in self.text_columns]
            # )
            # sentences: [sentence_1: str, sentence_2: str]
            sentences = [
                self.spacing(util.get_only_korean(item[text_column])) 
                for text_column in self.text_columns
            ]
            
            augment_sentence_1 = augmentation.EDA(sentences[0])
            augment_sentence_2 = augmentation.EDA(sentences[1])
            for s1, s2 in zip(augment_sentence_1, augment_sentence_2):
                text = s1 + '[SEP]' + s2
            
                outputs = self.tokenizer(text, add_special_tokens=True, padding='max_length', truncation=True)
                data.append(outputs['input_ids'])
        return data

    def preprocessing(self, data):
        data = data.drop(columns=self.delete_columns)

        # 텍스트 데이터를 전처리합니다.
        inputs = self.tokenizing(data)

        # 타겟 데이터가 없으면 빈 배열을 리턴합니다.
        augment_cnt = 3
        try:
            targets = data[self.target_columns].values.tolist()
            for target in data[self.target_columns].values.tolist():
                targets += [target for _ in range(augment_cnt+1)]
        except:
            targets = []
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
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=8)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset, batch_size=self.batch_size)
    
    
