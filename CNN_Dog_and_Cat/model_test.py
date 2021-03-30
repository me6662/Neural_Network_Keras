from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

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

my_model = load_model('my_model')

testing_data_generator = ImageDataGenerator(rescale=1./255)

test_set = testing_data_generator.flow_from_directory(
    'Dataset/PetImages/Test', target_size=(INPUT_SIZE, INPUT_SIZE), batch_size=BATCH_SIZE, class_mode='binary')

score = my_model.evaluate_generator(test_set, steps=len(test_set))

for idx, metric in enumerate(my_model.metrics_names):
    print("{} : {}".format(metric, score[idx]))
