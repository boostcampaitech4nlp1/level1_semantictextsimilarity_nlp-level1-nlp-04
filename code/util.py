import re

def get_only_korean(text: str):
    ''' 
        한글을 제외한 모든 문자 제거(특수 문자 포함) 
    '''
    return re.sub(r'[^\uAC00-\uD7A30-9a-zA-Z ]', '', text)