from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
import os
import random

"""
<이미지 증강>
이미지를 약간 변형해서 새로운 훈련 data 를 만드는일
이미지 회전/이동/좌우반전/확대 등의 방법
한정된 이미지 data를 이용해 대량의 data 생성 가능
이미지 증강을 통해 새로운 훈련 데이터셋을 인위적으로 만듬
"""

image_generator = ImageDataGenerator(
    rotation_range=30, width_shift_range=0.2, height_shift_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')


fig, ax = plt.subplots(2, 3, figsize=(20, 10))
all_images = []


_, _, dog_images = next(os.walk('Dataset/PetImages/Dog'))
# sample 은 list 에서 1개 뽑느데 idx[0] 만 리턴
random_img = random.sample(dog_images, 1)[0]
random_img = plt.imread('Dataset/PetImages/Dog/'+random_img)  # array 리턴됨
all_images.append(random_img)

random_img = random_img.reshape((1,)+random_img.shape)
# [1,2,3] 이라면 [[1,2,3]] 이 된다.
sample_agumented_images = image_generator.flow(random_img)

for _ in range(5):
    agumented_imgs = sample_agumented_images.next()  # 반복가능 개체의 다음 요소리턴
    for img in agumented_imgs:
        all_images.append(img.astype('uint8'))

for idx, img in enumerate(all_images):
    ax[int(idx/3), idx % 3].imshow(img)
    ax[int(idx/3), idx % 3].axis('off')

    if idx == 0:
        ax[int(idx/3), idx % 3].set_title('Original Image')
    else:
        ax[int(idx/3), idx % 3].set_title('Agumented Image {}'.format(idx))

plt.show()
