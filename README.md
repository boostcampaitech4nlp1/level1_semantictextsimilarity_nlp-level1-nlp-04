# **level1 NLP 4조**
## **[NLP] 문장 간 유사도 계산**

복수의 문장에 대한 유사도를 수치로 제시하는 NLP Task입니다.

정보 추출, QA 및 요악과 같은 NLP 문제에 널리 활용됩니다.

실제 어플리케이션에서는 데이터 증강, 질문 제안, 중복 문장 탐지 등에 응용되고 있습니다.

이번 Competition에서는 STS 데이터셋을 활용해 두 문장의 유사도를 측정하는 모델을 구성하는 것입니다.

<br><br><br>

## **Contributors**

|이름|id|역할|
|:--:|:--:|--|
|이용우|[@wooy0ng](https://github.com/wooy0ng)|협업 리딩, 데이터 전처리, 데이터 후처리|
|강혜빈|[@hyeb](https://github.com/hyeb)|모델링 (모델 튜닝, 결과 분석)
|권현정|[@malinmalin2](https://github.com/malinmalin2)|데이터 전처리, 데이터 후처리|
|백인진|[@eenzeenee](https://github.com/eenzeenee)|모델링 (모델 실험의 이론적 근거 마련)|
|이준원|[@jun9603](https://github.com/jun9603)|모델링 (모델 튜닝, 결과 분석, 앙상블)


<br><br><br>


## **Data**

기본 데이터
- train set : 9,324개
- validation set : 550개
- evaluation set : 1,110개

✓ 평가 데이터의 50%는 public 점수 계산에 반영되어 실시간 리더보드에 표기된다.

✓ 나머지 50%는 private 점수 계산에 반영되어 대회 종료 후 평가된다. 


<br><br><br>


## **Stacks**

|idx|experiment|  
|--|--|
|1|scaling up (model)|
|2|easy data augmentatiion(EDA)|
|3|back translation|
|4|bert mean pooling|
|5|ensemble|


<br><br><br>

## **project tree**

```
── README.md
└── code
    ├── inference.py
    ├── requirements.txt
    └── train.py
```

<br><br><br>

## **Train**

```
$ python main.py --augment [value]
```

### **augment**
- `--model_name` : huggingface model name (str)
- `--batcsh_size` : batch_size (int)
- `--max_epoch` : epoch_size (int)
- `--shuffle` : shuffle dataset (bool)
- `--learning_rate` : learning rate (float)
- `--train_path` : train dataset's path (str)
- `--dev_path` : validation dataset's path (str)
- `--test_path` : evaluation dataset's path (str)
- `--predict_path` : prediction dataset's path (str)

<br><br><br>

## **Inference**

```
$ python inference.py --augment [value]
```

### **augment**
- `--model_name` : huggingface model name (str)
- `--predict_path` : prediction dataset's path (str)


<br><br><br>


