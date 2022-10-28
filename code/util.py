import re
import os
import numpy as np

def get_only_korean(text: str):
    ''' 
        한글을 제외한 모든 문자 제거(특수 문자 포함) 
    '''
    return re.sub(r'[^\uAC00-\uD7A30-9a-zA-Z ]', '', text)

def npy_object_save(filename: str, arr: np.ndarray) -> None:
    if not os.path.exists('./object'):
        os.makedirs('object')
    np.save(f'./object/array-{filename}.npy', arr=arr)
    return

def npy_object_load(filename: str, arr: np.ndarray) -> np.ndarray:
    return np.load(file=f'./object/array-{filename}.npy')
    