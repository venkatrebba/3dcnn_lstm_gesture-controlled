import os
import tensorflow as tf
import time
import datetime
from logger_handler import Logger
from argparse import ArgumentParser
from dataGenerator import DataGenerator

os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Command Line argument parser.
parser = ArgumentParser(description='Gesture model train and evaluate')

# List of supported CL arguments.
required_args = parser.add_argument_group('Required Arguments')

# List of required CL arguments.
required_args.add_argument('-d', "--train",
                           help="train directory",
                           required=True)

required_args.add_argument("-e", "--eval",
                           help="eval directory",
                           required=True)

required_args.add_argument("-b", "--batch",
                           help="batch list",
                           required=True)

required_args.add_argument("-l", "--lr",
                           help="learning rate",
                           required=True)

required_args.add_argument("-ep", "--epochs",
                           help="Epochs steps",
                           required=True)

args = parser.parse_args()

# Train directory path
train_dir = args.train

# Eval directory path
val_dir = args.eval

batch_list = [int(num) for num in (args.batch).split(",")]
lr_list = [float(num) for num in (args.lr).split(",")]
epochs = int(args.epochs)

date_time = (datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
start_time = time.time()
logger = Logger('log_train_{}'.format(time.time()), "video_train_logs" + date_time + ".txt").build()

labels = os.listdir(train_dir)
labels.sort()

logger.info("train_dir: {}, val_dir:{}".format(train_dir, val_dir))

logger.info("\tLoading Training dataset................!!!\n")
generator = DataGenerator()
training_data, training_targets = generator.load_data(train_dir)

logger.info("\tLoading Testing dataset................!!!\n")
test_Xs, test_targets = generator.load_data(val_dir)

logger.info("Total time to compute features: \t{}".format(time.time() - start_time))
logger.info('Training dataset shape: {}, {} '.format(training_data.shape, training_targets.shape))
logger.info('Testing dataset shape: {}, {} '.format(test_Xs.shape, test_targets.shape))

# My model
class Conv3DModel(tf.keras.Model):
    def __init__(self):
        super(Conv3DModel, self).__init__()

        super(Conv3DModel, self).__init__()
        # Convolutions
        self.convloution1 = tf.compat.v2.keras.layers.Conv3D(32, (3, 3, 3), activation='relu', name="conv1",
                                                             data_format='channels_last', padding='SAME')
        self.pooling1 = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), data_format='channels_last', name="pool1")
        self.convloution2 = tf.compat.v2.keras.layers.Conv3D(64, (3, 3, 3), activation='relu', name="conv2",
                                                             data_format='channels_last', padding='SAME')
        self.pooling2 = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), data_format='channels_last', name="pool2")

        # LSTM & Flatten
        self.convLSTM = tf.keras.layers.ConvLSTM2D(40, (3, 3))
        self.flatten = tf.keras.layers.Flatten(name="flatten")

        # Dense layers
        self.d1 = tf.keras.layers.Dense(128, activation='relu', name="d1")
        self.dropout = tf.keras.layers.Dropout(rate=0.3)
        self.out = tf.keras.layers.Dense(len(labels), activation='softmax', name="output")

    def call(self, x):

        print("input.shape", x.shape)

        x = self.convloution1(x)
        print("conv1.shape", x.shape)

        x = self.pooling1(x)
        print("pool1.shape", x.shape)

        x = self.convloution2(x)
        print("conv2.shape", x.shape)

        x = self.pooling2(x)
        print("pool2.shape", x.shape)

        x = self.convLSTM(x)
        print("convLSTM.shape", x.shape)

        x = self.flatten(x)
        print("flatten.shape", x.shape)

        x = self.d1(x)
        print("d1.shape", x.shape)

        x = self.dropout(x)
        print("dropout.shape", x.shape)

        return self.out(x)


for ind, batch in enumerate(batch_list):

    logger.info("Batch:{} training.....!!!".format(batch))
    lr = lr_list[ind]

    model = Conv3DModel()

    # choose the loss and optimizer methods
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(lr=lr, epsilon=0.001),
                  metrics=['accuracy'])

    # include the epoch in the file name. (uses `str.format`)
    model_dir = "gesture_model/batch_{}/{}".format(batch, time.time())
    checkpoint_path = model_dir + "/weights.{epoch:02d}-{val_loss:.3f}"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, verbose=0, save_weights_only=False)

    logdir = model_dir + "/scalars/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.00001,
                                                           patience=8, restore_best_weights=True)

    # Run the training
    history = model.fit(training_data, training_targets,
                        callbacks=[cp_callback, tb_callback, early_stop_callback],
                        validation_data=(test_Xs, test_targets),
                        batch_size=batch,
                        epochs=epochs)

    # save the model for use in the application
    model.save_weights(model_dir + "/final_weights", save_format='tf')



