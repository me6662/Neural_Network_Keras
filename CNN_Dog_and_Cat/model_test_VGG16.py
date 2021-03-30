from keras.applications.vgg16 import VGG16
from keras.models import Model, load_model
from keras.layers import Dropout, Flatten, Dense
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

INPUT_SIZE = 128
BATCH_SIZE = 16
STEPS_PER_EPOCH = 200
EPOCHS = 3

testing_data_generator = ImageDataGenerator(rescale=1./255)

test_set = testing_data_generator.flow_from_directory(
    'Dataset/PetImages/Test/', target_size=(INPUT_SIZE, INPUT_SIZE), batch_size=BATCH_SIZE, class_mode='binary')

my_vgg16 = load_model('my_model_vgg16')

score = my_vgg16.evaluate_generator(test_set, steps=len(test_set))

for idx, metric in enumerate(my_vgg16.metrics_names):
    print("{} : {}".format(metric, score[idx]))
