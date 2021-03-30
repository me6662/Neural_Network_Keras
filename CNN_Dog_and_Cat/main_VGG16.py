"""
전이학습
- 특정 대상을 예측하게 훈련시킨 모델을 다른 대상도 예측할 수 있게 바꾸는 머신러닝 기법
예) 승용차를 분류하는 모델을 약간 수정하여 트럭을 분류하는데 사용함

- CNN 에서는 Conv - Maxpool 레이어를 고정하고 마지막  Dense Layer 만 다시 학습시키면
전이학습을 구현할 수 있다.

- 왜냐면 Conv-Maxpool 레이어 조합은 이미지내 특징을 찾아내는데, 사물의 생김새가 일부분 비슷 하다면, 
이 레이어도 재사용 할 수 있다.
즉 슨, 완전연결레이어만 다시 훈련시켜 새로운 클래스를 예측하도록 바꾸면 된다. 
단 A,B가 가급적 유사한 사물이어야 할것이다.

여기서는 이미 만들어진 모델인 VGG16을 사용해서 개 와 고양이를 분류해본다.
VGG16은 대회에 나왔던 분류모델로써, 1000개의 클래스를 분류한다. 여기는 개랑 고양이도 포함되어 있다.
VGG16은 케라스에 내장되어 있으니까 바로 쓸수있다.
"""

from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dropout, Flatten, Dense
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

INPUT_SIZE = 128
BATCH_SIZE = 16
STEPS_PER_EPOCH = 200
EPOCHS = 3

vgg16 = VGG16(include_top=False, weights='imagenet',
              input_shape=(INPUT_SIZE, INPUT_SIZE, 3))

# include_top = False 면 마지막에 Dense 레이어를 로드 하지 않는다.

# 우리는 VGG16은 학습 안시킬것이다.
for layer in vgg16.layers:
    layer.trainable = False


# VGG16은 Sequential 을 쓰지 않아 사용법이 약간 다르다.
# 케라스에 직접 레이어를 추가하는 방식 ,, Sequential 은 이를 단순화 한거
input_ = vgg16.input
output_ = vgg16(input_)
last_layer = Flatten(name='flatten')(output_)
last_layer = Dense(1, activation='sigmoid')(last_layer)

model = Model(inputs=input_, outputs=last_layer)


# compile


model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])

training_data_generator = ImageDataGenerator(rescale=1./255)

training_set = training_data_generator.flow_from_directory(
    'Dataset/PetImages/Train/', target_size=(INPUT_SIZE, INPUT_SIZE), batch_size=BATCH_SIZE, class_mode='binary')

# fit
model.fit_generator(
    training_set, steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS, verbose=1)

model.save("my_model_vgg16")
