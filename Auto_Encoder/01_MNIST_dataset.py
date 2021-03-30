from matplotlib import pyplot as plt
from keras.datasets import mnist
"""
mnist

손으로 쓴 숫자 7만개 (28x28)
숫자가 하나만 그려졌고 이 숫자가 무엇인지에 대한 레이블 정보 포함
"""
training_set, testing_set = mnist.load_data()
X_train, y_train = training_set
X_test, y_test = testing_set


# Matplotlib 확인
fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10)
      ) = plt.subplots(2, 5, figsize=(10, 5))


for idx, ax in enumerate([ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10]):
    for i in range(1000):
        if y_test[i] == idx:
            ax.imshow(X_test[i], cmap='gray')
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            break


plt.tight_layout()
plt.show()
