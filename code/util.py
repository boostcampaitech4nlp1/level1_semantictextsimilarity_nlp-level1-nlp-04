import re
import os
import numpy as np

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


def npy_object_save(filename: str, arr: np.ndarray) -> None:
    '''
        ### file name: array-[filename].npy
    '''
    if not os.path.exists('./object'):
        os.makedirs('object')
    np.save(f'./object/array-{filename}-data.npy', arr=arr)
    return

def npy_object_load(filename: str) -> np.ndarray:
    '''
        ### file name: array-[filename].npy
    '''
    print("load train dataset object...")
    return np.load(file=f'./object/array-{filename}-data.npy')
    