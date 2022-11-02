import re
import os
import numpy as np
import pandas as pd

import pickle as pkl

def get_custom_data():
    filename = './data/new_train.csv'
    df = pd.read_csv(filename)
    df.drop(['sentence_1', 'sentence_2'], axis=1, inplace=True)
    df.rename(columns={
        'new_sentence_1': 'sentence_1',
        'new_sentence_2': 'sentence_2'
    }, inplace=True)
    
    tmp1 = df[df['label']!=0]
    tmp2 = df[df['label']==0].head(500)
    data = pd.concat([tmp1, tmp2])
    return data

def get_usable_char(text: str):
    ''' 
        일반적이지 않은 특수 문자 제거 
    '''
    text = text.strip()
    return re.sub(r'[^\uAC00-\uD7A30-9a-zA-Z\s\!\%\?\~]', '', text)

def get_move_end_usable_symbol(text: str):
    '''
        문장의 느낌에 영향을 줄 가능성이 높은 특수문자와 문장 띄어쓰기
    '''
    pattern = r'[\!\?\~]'
    symbol = ''.join(re.findall(pattern, text))
    text = re.sub(pattern, '', text)
    return ''.join([text, symbol])

def get_only_korean(text: str):
    ''' 
        한글을 제외한 모든 문자 제거(특수 문자 포함) 
    '''
    return re.sub(r'[^\uAC00-\uD7A30-9a-zA-Z\s]', '', text)

def pkl_object_save(file: str, obj) -> None:
    '''
        ### file: input file path
    '''
    print("save object...")
    if not os.path.exists('./object'):
        os.makedirs('./object')
    
    with open(file+'.pkl', 'wb+') as f:
        pkl.dump(obj, f)
    return

def pkl_object_load(file: str) -> np.ndarray:
    '''
        ### file: input file path
    '''
    print("load object...")
    obj = None
    with open(file+'.pkl', 'rb+') as f:
        obj = pkl.load(f)
    return obj