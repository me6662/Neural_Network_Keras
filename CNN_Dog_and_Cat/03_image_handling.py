from utils import train_test_split
"""
메모리 문제없이 대량의 이미지를 학습하기 위해 전처리 작업을 수행해줌
이를 위해서는 디렉터리도 정해진 폴더구조로 만들어야 되는데 utils.py 에다가 복사 조짐 그냥
"""

src_folder = 'Dataset/PetImages/'
train_test_split(src_folder)
