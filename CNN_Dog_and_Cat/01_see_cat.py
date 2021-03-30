from matplotlib import pyplot as plt
import os
import random

# 파일명 리스트  가져오기
_, _, cat_images = next(os.walk('Dataset/PetImages/Cat'))
"""
next의 [0]번 결과는 특정 디렉토리의 위치 값을 저장
next의 [1]번 결과는 특정 디렉토리의 모든 하위 디렉토리 리스트를 저장
next의 [2]번 결과는 특정 디렉토리에 포함된 모든 파일 리스트를 저장
"""

# subplot 준비
fig, ax = plt.subplots(3,3, figsize=(20,10))


# 무작위 이미지 차트 구성
for idx, img in enumerate(random.sample(cat_images, 9)) :
    img_read = plt.imread('Dataset/PetImages/Cat/'+img)
    ax[int(idx/3), idx%3].imshow(img_read)
    ax[int(idx/3), idx%3].axis('off')
    ax[int(idx/3), idx%3].set_title('Cat/'+img)

plt.show()