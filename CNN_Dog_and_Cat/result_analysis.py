import random
from matplotlib import pyplot as plt
from keras.applications.vgg16 import VGG16
from keras.models import Model, load_model
from keras.layers import Dropout, Flatten, Dense
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

INPUT_SIZE = 128
BATCH_SIZE = 16
STEPS_PER_EPOCH = 200
EPOCHS = 3

"""
마지막에 Sigmoid 를 쓰므로, 

크게 맞은 결과 : 모델이 예측한 값이 0.8초과 0.2 미만 이고, 맞춘 경우
크게 틀린 결과 : 모델이 예측한 값이 0.8초과 0.2 미만 이고, 틀린 경우
근소하게 틀린 결과 : 모델이 예측한 값이 0.4 에서 0.6 사이이고, 틀린 경우
"""

# 시각화 데이터 생성

testing_data_generator = ImageDataGenerator(rescale=1./255)

test_set = testing_data_generator.flow_from_directory(
    'Dataset/PetImages/Test/', target_size=(INPUT_SIZE, INPUT_SIZE), batch_size=1, class_mode='binary')  # batchsize = 1;;


strongly_wrong_idx = []
strongly_right_idx = []
weakly_wrong_idx = []

model = load_model('my_model_vgg16')


for i in range(test_set.__len__()):
    img = test_set.__getitem__(i)[0]
    pred_prob = model.predict(img)[0][0]  # sigmoid 통과전값

    pred_label = int(pred_prob > 0.5)  # 예측값
    actual_label = int(test_set.__getitem__(i)[1][0])  # 실제 값

    if pred_label != actual_label and (pred_prob > 0.8 or pred_prob < 0.2):
        strongly_wrong_idx.append(i)
    elif pred_label != actual_label and (pred_prob > 0.4 or pred_prob < 0.6):
        weakly_wrong_idx.append(i)
    elif pred_label == actual_label and (pred_prob > 0.8 or pred_prob < 0.2):
        strongly_right_idx.append(i)

    # 이미지를 적당히 가져왔으면 멈춘다.
    if (len(strongly_wrong_idx) >= 9) and (len(weakly_wrong_idx) >= 9) and (len(strongly_right_idx) >= 9):
        break


# 분류한 이미지 결과 중 거른 것중 무작위 9개를 표시해보자.

def plot_on_grid(test_set, idx_to_plot, img_size=INPUT_SIZE):
    fig, ax = plt.subplots(3, 3, figsize=(20, 10))

    for i, idx in enumerate(random.sample(idx_to_plot, 9)):
        img = test_set.__getitem__(idx)[0].reshape(img_size, img_size, 3)
        ax[int(i/3), i % 3].imshow(img)
        ax[int(i/3), i % 3].axis('off')


# 크게맞은 결과 중 아홉게
plot_on_grid(test_set, strongly_right_idx)
plt.show()
