# level1_semantictextsimilarity_nlp-level1-nlp-04

## **개요**

복수의 문장에 대한 유사도를 수치로 제시하는 NLP Task입니다.

정보 추출, QA 및 요악과 같은 NLP 문제에 널리 활용됩니다.

실제 어플리케이션에서는 데이터 증강, 질문 제안, 중복 문장 탐지 등에 응용되고 있습니다.

이번 Competition에서는 STS 데이터셋을 활용해 두 문장의 유사도를 측정하는 모델을 구성하는 것입니다.

<br><br><br><br>

### **Scaling Up**

모델의 성능을 끌어올리기 위해서 가장 먼저 할 수 있는 작업입니다.

2017년, Transformer의 발표 이후, 대부분의 언어 모델은 Transformer를 기반으로 만들어지고 있는데

언어 모델의 성능 개선을 위해서 먼저 모델의 Scale을 크게 늘리는 방법을 사용하였습니다.

언어 모델의 파라미터를 엄청나게 키운 모델인 GPT-3의 performance로

모델의 크기를 키울 수록 성능이 높아진다는 것은 어느정도 기정사실화되었습니다.


![image](https://user-images.githubusercontent.com/37149278/204948770-99413bd7-a484-4c7f-9842-c4aaeafc6021.png)


klue의 roberta-small과 roberta-large 모델을 동일 데이터셋으로 실험해보았을 때

모델의 크기가 성능에 큰 영향을 미친다는 것을 알 수 있었습니다.

Scale이 큰 여러가지 모델을 사용해볼 수 있습니다.
 (koelectra-base, xlm-roberta-large, funnel-kor-base 등)


<br><br><br><br>

### **Data Augmentation**

이번 Competition에서 데이터의 개수는 대략 9000개 정도로 작은 편에 속합니다.

때문에 데이터를 증강하는 방법을 어렵지 않게 떠올릴 수 있습니다.

NLP에서 Data Augmentation은 여러가지가 있는데

크게 아래와 같이 5가지 Augmentation 방법이 있습니다.


|방법|내용|
|:---:|:---:|
|SR (Synonym Replacement)|WordNet을 참고하여 특정 단어를 유의어로 교체 <br>(WordNet의 Quality에 크게 영향을 받음)|
|RI (Random Insertion)|임의의 단어 삽입<br>(기존 문장의 의미와 상반된 단어가 삽입 될 수 있음)|
|RD (Random Deletion)|임의의 단어 삭제<br>(문장 내에서 중요한 역할을 하는 단어가 삭제 될 수 있음)|
|RS (Random Shuffle)|단어의 위치 변환<br>(한국어의 경우 문장 내 단어의 위치가 바뀌어도 의미가 변하지 않는다는 특징을 이용)|
|BT (Back Translation)|역번역<br>(잘 학습된 기계번역기를 이용해 한국어를 영어로 번역 후, 다시 한국어로 재번역)|



위에서 두 가지 방법을 사용하여 데이터를 증강시켜 학습해보았습니다.
- RS(Random Shuffle), BT(Back Translation)
![image](https://user-images.githubusercontent.com/37149278/204950180-849adf6d-e737-4189-93e1-d1a99a178da7.png)

Back Translation의 경우 학습이 잘 되지 않았는데 그 이유는 아래와 같이 생각해보았습니다.
> 역번역의 경우 기계번역기의 성능에 크게 의존하는 방법이다.
>
> 역번역 후 의미가 변질된다면 학습에 악영향을 미칠 것이다.

<br><br><br>

Random Shuffle로 데이터를 증강시켜 학습해보았습니다.
<br>

![image](https://user-images.githubusercontent.com/37149278/204950928-54d7eb91-570e-43f8-8fa1-0472d340aceb.png)

<br>

RS의 경우 학습 후 성능이 미세하게 올랐습니다.

그 이유는 아래와 같이 추측해보았습니다.

> pretrained된 모델은 한국어 문법을 모두 준수하는 데이터를 사용하였다.
>
> 우리가 사용하는 데이터는 구어체이기 때문에 문법이 맞지 않는 경우가 매우 많다.
> 
> 단어의 위치를 바꿈으로써 문법은 맞지 않지만 의미가 같은 문장을 사용했기 때문에 성능 향상이 되었다고 추측하였다.


<br><br><br>


### **BERT Mean Pooling**
BERT로부터 각 단어에 대한 Output Embedding 벡터를 어떻게 활용하는 지에 대한 방법은 여러가지가 있습니다.

먼저 분류를 위한 [CLS] 토큰의 출력 벡터를 새로운 Embedding 벡터로 간주하는 방법이 있습니다.
![image](https://user-images.githubusercontent.com/37149278/204951658-4fc3ce17-202d-4ff1-a479-2c2b21d39e56.png)

하지만 이는 [CLS] 토큰을 입력 문장에 대한 summarize된 표현이라고 가정하는 것입니다.

물론 [CLS] 토큰만 활용할 수도 있지만 항상 [CLS] 토큰이 입력 문장 전체를 대변하지는 않을 수도 있을 것입니다.

때문에 아래와 같이 입력으로 들어간 모든 토큰들에 대한 출력 벡터를 평균내어

Embedding 벡터를 만들어 내는 Mean Pooling 방법을 사용해보았습니다.

![image](https://user-images.githubusercontent.com/37149278/204951955-b06fa357-60a7-4ea8-8bd5-84daa03e8b0d.png)


<br><br>

실제로 이번 Competition에서 Mean pooling을 적용하지 않은 모델보다 좋은 성능을 보여주었습니다.
![image](https://user-images.githubusercontent.com/37149278/204952003-ad1c41cd-b350-4f1a-aa4e-3015580af060.png)



<br><br><br>

# 사용 방법

### Train

```bash
python main.py
```

### Inference
```bash
python inference
```


<br><br><br>