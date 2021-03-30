from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

model = Sequential()

# Parameter
FILTER_SIZE = 3  # 보통 3 x 3 필터를 많이 사용함
NUM_FILTERS = 32  # 특성을 캐치할 필터의 수 (weight 이라고 봐도됨) 훈련속도와, 예측성능을 고려
INPUT_SIZE = 32  # 데이터셋 이미지가 다르지만, 일정하게 하여 훈련시간을 줄인다.
MAXPOOL_SIZE = 2  # 최대풀링크기 : 일반적으로  2X2 로 지정하여 차원을 반으로 줄인다.

BATCH_SIZE = 16  # 일괄처리량 : Gradient Descent 를 할때 한번에 처리되는 데이터 기준수
# 16개면 16개로 수행, 큰값일 수록 정확히 훈련시키지만 시간과 메모리를 많이 소모

STEPS_PER_EPOCH = 20000/BATCH_SIZE  # 에폭당 반복수 : 훈련데이터셋 개수 / batch size
# Weight와 Bias를 1회 업데이트하는 것을 1 Step이라 부른다.
EPOCHS = 10  # 에폭 : 전체 데이터셋 반복 횟수


# --- model ---
# conv-maxpool-conv-maxpool - Dense - Dense

model.add(Conv2D(NUM_FILTERS, (FILTER_SIZE, FILTER_SIZE),
                 input_shape=(INPUT_SIZE, INPUT_SIZE, 3), activation='relu'))  # input shape : (32,32,3)  3은 rgb 값
model.add(MaxPooling2D(pool_size=(MAXPOOL_SIZE, MAXPOOL_SIZE)))

model.add(Conv2D(NUM_FILTERS, (FILTER_SIZE, FILTER_SIZE), activation='relu'))
model.add(MaxPooling2D(pool_size=(MAXPOOL_SIZE, MAXPOOL_SIZE)))

# conv - pool 레이어를 거친 후 나온 3 차원 벡터는 Dense Layer 에 넣어줄려면 1차원 벡터로 변경시켜야 한다.
# flatten() 레이어를 사용한다. (5,5,3) > (75) 로 변경
# 이렇게 해도 이미지 분류가 되는 이유는 어차피 학습 완료후 모델에 입력을 넣어줘도 똑같이 flatten 할것 이기 때문이다.

model.add(Flatten())

model.add(Dense(units=128, activation='relu'))


# drop out Layer
# 입력값 중 일정 비율을 무작위로 골라 0으로 만든다.
# 모델이 특정 가중치에 너무 치중하는 걸 막아 과적합 을 줄이는 효과가 있다.

model.add(Dropout(0.5))

# 마지막 결과 리턴 레이어
# 개 / 고양이 분류 이므로! sigmoid
model.add(Dense(units=1, activation='sigmoid'))


# ---- Optimize ----
# CNN 훈련의 경우 adam 을 가장 많이 쓰고 Loss 함수는 binary_crossentropy를 사용한다. (이진분류)
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])

# ---- fit ----
# Image DataGenerator 의 flow from directory 메서드를 통해 실시간으로 모델을 훈련시킨다.(필요한 만큼만 데이터셋을 메모리에 로드)

training_data_generator = ImageDataGenerator(rescale=1./255)  # 이미지 증강

training_set = training_data_generator.flow_from_directory(
    'Dataset/PetImages/Train/', target_size=(INPUT_SIZE, INPUT_SIZE), batch_size=BATCH_SIZE, class_mode='binary')

model.fit_generator(
    training_set, steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS, verbose=1)


model.save("my_model")
