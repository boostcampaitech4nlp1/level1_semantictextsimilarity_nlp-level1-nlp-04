from json import JSONDecodeError
import re
import pytorch_lightning as pl
import torch
import transformers
from tqdm.auto import tqdm
import pandas as pd
import numpy as np

# from pykospacing import Spacing
from hanspell import spell_checker

import os
import util
import augmentation
import sys
import random

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
    def __init__(self, 
                 model_name, 
                 batch_size, 
                 shuffle, 
                 train_path, 
                 dev_path, 
                 test_path, 
                 predict_path,
                 augmentation,
                 num_aug
        ):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.predict_path = predict_path
        
        self.augmentation = augmentation
        self.num_aug = num_aug

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None
        
        # self.spacing = Spacing()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, max_length=160)
        
        # main data
        self.target_columns = ['label', 'binary-label']
        self.delete_columns = ['id']
        self.text_columns = ['sentence_1', 'sentence_2']
        

    def tokenizing(self, dataframe, stage=None):
        data = []
        for idx, item in tqdm(dataframe.iterrows(), desc='tokenizing', total=len(dataframe)):
            if stage == 'fit':
                try:
                    sentences = [
                        util.get_usable_char(spell_checker.check(item[text_column]).checked)
                        for text_column in self.text_columns
                    ]
                except BaseException as e:
                    # 띄어쓰기가 전혀 없는 이상한 문장에 대해서 에러 발생
                    sentences = [
                        util.get_usable_char(item[text_column])
                        for text_column in self.text_columns
                    ]
                
                if self.augmentation:
                    if len(sentences[0]) > 1 and len(sentences[1]) > 1:
                        augment_sentence_1 = augmentation.EDA(sentences[0], num_aug=self.num_aug)
                        augment_sentence_2 = augmentation.EDA(sentences[1], num_aug=self.num_aug)
                    else:
                        augment_sentence_1 = [sentences[0]] * (self.num_aug+1)
                        augment_sentence_2 = [sentences[1]] * (self.num_aug+1)
                    
                    for s1, s2 in zip(augment_sentence_1, augment_sentence_2):
                        text = s1 + '[SEP]' + s2
                        outputs = self.tokenizer(text, add_special_tokens=True, padding='max_length', truncation=True)
                        data.append(outputs['input_ids'])
                else:
                    text = '[SEP]'.join(
                        [item[text_column] for text_column in self.text_columns]
                    )
                    outputs = self.tokenizer(text, add_special_tokens=True, padding='max_length', truncation=True)
                    data.append(outputs['input_ids'])
            else:
                text = '[SEP]'.join(
                    [util.get_only_korean(item[text_column]) for text_column in self.text_columns]
                )
                outputs = self.tokenizer(text, add_special_tokens=True, padding='max_length', truncation=True)
                data.append(outputs['input_ids'])    
        return data

    def preprocessing(self, data, stage, stage_type=''):
        data = data.drop(columns=self.delete_columns)
        if stage_type == 'train':
            tmp1 = data[data['label']!=0]
            tmp2 = data[data['label']==0].sample(500)
            tmp3 = util.get_custom_data()
            data = pd.concat([tmp1, tmp2, tmp3])
        
        # 텍스트 데이터를 전처리합니다.
        if self.augmentation:
            filename = f'./object/{stage_type}-aug-data.npy'
        else:
            filename = f'./object/{stage_type}-data.npy'
            
        if os.path.exists(filename):
            inputs = util.npy_object_load(filename).tolist()
        else:
            inputs = self.tokenizing(data, stage)
        # inputs = self.tokenizing(data, stage)

        # 타겟 데이터가 없으면 빈 배열을 리턴합니다.
        targets = []
      
        if self.augmentation:
            filename = f'./object/{stage_type}-aug-target.npy'
        else:
            filename = f'./object/{stage_type}-target.npy'
            
        if os.path.exists(filename):
            targets = util.npy_object_load(filename).tolist()
        else:
            if stage == 'fit':
                if self.augmentation:
                    try:
                        limit = 0.2
                        for label, binary_label in data[self.target_columns].values.tolist():
                            if label != 0:  # -0.2 ~ -0.1
                                rands = [(round(label-random.uniform(-limit, -0.1), 1), binary_label)
                                            if label-limit > 0 else (label, binary_label) 
                                            for _ in range(self.num_aug)
                                        ]
                                targets += [(label, binary_label)] + rands
                            else:
                                targets += [(label, binary_label) for _ in range(self.num_aug+1)]
                    except:
                        targets = []
                else:
                    try:
                        targets = list(map(tuple, data[self.target_columns].values.tolist()))
                    except:
                        targets = []
            else:
                try:
                    targets =  list(map(tuple, data[self.target_columns].values.tolist()))
                except:
                    targets = []
        return inputs, targets
    
    def setup(self, stage='fit'):
        if stage == 'fit':
            # 학습 데이터와 검증 데이터셋을 호출합니다
            train_data = pd.read_csv(self.train_path)
            val_data = pd.read_csv(self.dev_path)

            # 학습데이터 준비
            stage_type='train'
            train_inputs, train_targets = self.preprocessing(train_data, stage, stage_type=stage_type)
            
            PATH = './object/'
            if self.augmentation:
                util.npy_object_save(PATH+stage_type+'-aug-data', np.asarray(train_inputs))    
                util.npy_object_save(PATH+stage_type+'-aug-target', np.asarray(train_inputs))    
            else:
                util.npy_object_save(PATH+stage_type+'-data', np.asarray(train_inputs))
                util.npy_object_save(PATH+stage_type+'-target', np.asarray(train_inputs))    

            # 검증데이터 준비
            val_inputs, val_targets = self.preprocessing(val_data, stage)

            # train 데이터만 shuffle을 적용해줍니다, 필요하다면 val, test 데이터에도 shuffle을 적용할 수 있습니다
            self.train_dataset = Dataset(train_inputs, train_targets)
            self.val_dataset = Dataset(val_inputs, val_targets)
        else:
            # 평가데이터 준비
            test_data = pd.read_csv(self.test_path)
            test_inputs, test_targets = self.preprocessing(test_data, stage)
            self.test_dataset = Dataset(test_inputs, test_targets)

            predict_data = pd.read_csv(self.predict_path)
            predict_inputs, predict_targets = self.preprocessing(predict_data, stage)
            self.predict_dataset = Dataset(predict_inputs, [])

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=8)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset, batch_size=self.batch_size)
    
    
