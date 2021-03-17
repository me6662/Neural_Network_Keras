from token import OP
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Embedding, LSTM
from keras.layers import Dense
from tensorflow.python.keras.backend import sigmoid
from matplotlib import pyplot as plt
# keras : 2.4.3
# numpy : 1.19.2
# tensorflow : 2.4.0 / cuda : 11.0 / cuDNN : 8.0.4


training_set, testing_set = imdb.load_data(num_words=10000)
# 데이터 셋에 저장된 순서대로 단어 10000 개를 가져옴 > 리뷰에 고유한 단어가 10000개만 쓰임.
X_train, y_train = training_set
X_test, y_test = testing_set

# 영화리뷰는 길이가 다 다른데, 신경망은 정해진 벡터만 입력받을 수 있다.

# 제로 패딩 : 벡터의 max 길이를 정해놓고 이거보다 짧은 벡터의 나머지 element 는 0으로 채운다. 긴벡터의 경우 maxlen 에 맞춰서 자름
# ex) maxlen = 10 , I love you > [1,2,3,4] > [1,2,3,4,0,0,0,0,0,0]

X_train_padded = sequence.pad_sequences(X_train, maxlen=100)
X_test_padded = sequence.pad_sequences(X_test, maxlen=100)


def train_model(Optimizer, X_train, y_train, X_val, y_val):
    # 모델 아키텍쳐
    # 입력 > 단어임베딩 레이어 > LSTM > 밀집레이어 > 출력 (0: 부정, 1 : 긍정)
    model = Sequential()
    # 전체 모델은 sequential 모델

    model.add(Embedding(input_dim=10000, output_dim=128))
    # 임베딩 레이어
    # input_dim : 임베딩 레이어의 입력벡터 차원 이는 데이터 셋에 있는 단어의 고유 갯수 와 동일해야함.
    # output_dim : 임베딩 레이어의 출력벡터 차원 , 실제로는 신중히 결정하지만 128 로 하자.

    model.add(LSTM(units=128))
    # LSTM 레이어
    # units : LSTM 의 유닛 갯수를 지정해준다. 유닛이 많을 수록 모델의 복잡도가 증가며, 이에 따라 훈련시간이 길어지고 , 과적합이 발생할 가능성이 높다.
    # activation : 셀상태와 은닉상태에 사용할 활성화 함수를 지정함. (default : tanh)
    # recurrent_activation : 망각게이트, 입력게이트, 출력게이트에 적용할 활성화 함수를 지정 (default : sigmoid) > keras 에서는 각각 다르게는 지정 불가

    model.add(Dense(units=1, activation=sigmoid))
    # Dense 레이어
    # 마지막은 단순한 밀집레이어로 0 , 1 결과를 내보내게 함

    # model.summary()
    # 모델 구조 검증

    # 컴파일

    # 손실함수 : 일반적으로 목표 변수가 이진변수 라면 binary_crossentropy 사용, 다중 클래스라면 categorial_crossentropy 사용
    # 옵티마이저 : LSTM 에는 100% 맞는 것은 없고 데이터셋에 따라 성능이 달라짐, 간혹 경사폭증문제로 인해 제대로 학습 못하는 경우가 발생함.
    #               가장좋은 방법은 옵티마이저 여러개 써보고 판단조지는거. (sgd,  RMSprop, adam 을 비교한다.)

    model.compile(loss='binary_crossentropy',
                  optimizer=Optimizer, metrics=['accuracy'])

    # 학습
    scores = model.fit(x=X_train, y=y_train, batch_size=128,
                       epochs=10, validation_data=(X_val, y_val), verbose=10)  # verbose : 학습과정을 자세히 보여줘라 라는 뜻
    return scores, model


# Optimizer 별 비교
# SGD_score, SGD_model = train_model(
#     Optimizer='sgd', X_train=X_train_padded, y_train=y_train, X_val=X_test_padded, y_val=y_test)
RMSprop_score, RMSprop_model = train_model(
    Optimizer='RMSprop', X_train=X_train_padded, y_train=y_train, X_val=X_test_padded, y_val=y_test)
#        Adam_score, Adam_model=train_model(
#     Optimizer='adam', X_train=X_train_padded, y_train=y_train, X_val=X_test_padded, y_val=y_test)


# 결과 분석!

plt.plot(range(1, 11),
         RMSprop_score.history['accuracy'], label='Training Accuracy')
plt.plot(range(1, 11),
         RMSprop_score.history['val_accuracy'], label='Validation Accuracy')
plt.axis([1, 10, 0, 1])  # 축 범위 (x 축 : 1~10, y축 : 0~ 1)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Train and Validation Accuracy using SGD Optimizer')
plt.legend()  # 색인
plt.show()


# 결과 sgd 는 학습 50% 도 못맞춤
# rmsprop ,adam 은 잘 맞추지만, adam 의 경우 학습 데이터 정확도와 테스트 데이터 정확도가 차이가 많이남 (과적합이란 소리)
# rmsprop 은 거의 비슷하게 감 -> 매우적합!
