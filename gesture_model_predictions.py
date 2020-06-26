
import tensorflow as tf

import tensorflow as tf
from argparse import ArgumentParser
from statistics import mode
from collections import Counter
from collections import OrderedDict
from logger_handler import Logger
from datetime import datetime
import matplotlib.image as img
import numpy as np
import os
import librosa
import cv2


# Command Line argument parser.
parser = ArgumentParser(description='Gesture model prediction code')

# List of supported CL arguments.
required_args = parser.add_argument_group('Required Arguments')

# List of required CL arguments.
required_args.add_argument('-m', "--model",
                           help="Gesture model directory",
                           required=True)

required_args.add_argument('-d', "--test_dir",
                           help="Testing directory",
                           required=False)

required_args.add_argument('-i', "--image_file",
                           help="Image file",
                           required=False)
# Input arguments
args = parser.parse_args()

# Train directory path
model_dir = args.model

# Test data directory
test_dir = args.test_dir

# single audio file
img_file = args.image_file

test_data_list = []


date_time = (datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
logger_file = "audio_logs_"+ date_time +".txt"

if img_file is not None:
    test_data_list.append(img_file)


if test_dir is not None:
    for (dirpath, dirnames, filenames) in os.walk(test_dir):
        test_data_list += [os.path.join(dirpath, file) for file in filenames]

test_data_list.sort()

classes = ['glitch', 'good']
logger = Logger('log_train', "audio_test_logs" + date_time + ".txt").build()

# My model
class Conv3DModel(tf.keras.Model):
    def __init__(self):
        super(Conv3DModel, self).__init__()
        # Convolutions
        self.conv1 = tf.compat.v2.keras.layers.Conv3D(32, (3, 3, 3), activation='relu', name="conv1", data_format='channels_last', padding='SAME')
        self.pool1 = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), data_format='channels_last', name="pool1")
        self.conv2 = tf.compat.v2.keras.layers.Conv3D(64, (3, 3, 3), activation='relu', name="conv2", data_format='channels_last', padding='SAME')
        self.pool2 = tf.keras.layers.MaxPool3D(pool_size=(2, 2,2), data_format='channels_last', name="pool2")

        # LSTM & Flatten
        self.convLSTM = tf.keras.layers.ConvLSTM2D(40, (3, 3))
        self.flatten = tf.keras.layers.Flatten(name="flatten")

        # Dense layers
        self.d1 = tf.keras.layers.Dense(128, activation='relu', name="d1")
        self.out = tf.keras.layers.Dense(len(classes), activation='softmax', name="output")


    def call(self, x):

        #print("input.shape", x.shape)

        x = self.conv1(x)
        #print("x.shape", x.shape)

        x = self.pool1(x)
        #print("x.shape", x.shape)

        x = self.conv2(x)
        #print("x.shape", x.shape)

        x = self.pool2(x)
        #print("x.shape", x.shape)

        x = self.convLSTM(x)
        #print("x.shape", x.shape)

        #x = self.pool2(x)
        #x = self.conv3(x)
        #x = self.pool3(x)
        x = self.flatten(x)
        #print("x.shape", x.shape)

        x = self.d1(x)
        #print("x.shape", x.shape)

        return self.out(x)


# %%
new_model = Conv3DModel()
# %%
new_model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(lr=0.00001, epsilon=0.001)
                  )

# %%
new_model.load_weights(os.path.join(model_dir,'final_weights'))


# Resize frames
def resize_image(image):
    image = img.imread(image)
    image = cv2.resize(image, (64, 64))
    return image


def preprocess_image(img):
    img = resize_image(img)
    return img



file_label_list = []
for image_file in test_data_list:
    logger.info('File: {}'.format(image_file))
    audio_feats = preprocess_image(image_file)

    predict = new_model.predict(audio_feats)
    #print("predict", predict)

    # Maximum probability in given predictions
    y_pred = np.argmax(predict, 1)
    logger.info("precition: ", classes[y_pred])


print("\n\n")
logger.info("Total summary ")
files_map = Counter(file_label_list)
for l,c in files_map.items():
    logger.info("Label: {}, Files : {}".format(l, str(c)))
print("\nExiting!")


# # %%
# to_predict = []
# num_frames = 0
# cap = cv2.VideoCapture(0)
# classe = ''
#
# while (True):
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#
#     # Our operations on the frame come here
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     to_predict.append(cv2.resize(gray, (64, 64)))
#
#     if len(to_predict) == 30:
#         frame_to_predict = np.array(to_predict, dtype=np.float32)
#         frame_to_predict = normaliz_data(frame_to_predict)
#         # print(frame_to_predict)
#         predict = new_model.predict(frame_to_predict)
#         classe = classes[np.argmax(predict)]
#
#         print('Classe = ', classe, 'Precision = ', np.amax(predict) * 100, '%')
#
#         # print(frame_to_predict)
#         to_predict = []
#         # sleep(0.1) # Time in seconds
#         # font = cv2.FONT_HERSHEY_SIMPLEX
#     cv2.putText(frame, classe, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
#
#     # Display the resulting frame
#     cv2.imshow('Hand Gesture Recognition', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()
