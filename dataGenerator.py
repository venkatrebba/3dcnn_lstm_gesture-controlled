import numpy as np
import progressbar
from multiprocessing import Pool
import cv2
import os
import matplotlib.image as img

class DataGenerator(object):

    def __init__(self):
        super().__init__()

    # Resize frames
    def resize_image(self, image):
        image = img.imread(image)
        image = cv2.resize(image, (64, 64))
        return image


    def preprocess_image(self, img):
        img = self.resize_image(img)
        return img

    def load_data(self, data_repo):
        X = []
        y = []

        pool = Pool()
        labels = os.listdir(data_repo)
        labels.sort()
        for ys, label in enumerate(labels):
            widgets = ['{}:'.format(label), progressbar.Bar(), progressbar.Percentage(), ' ', '', ' ',
                       progressbar.ETA(), ' ', ' ']
            pbar = progressbar.ProgressBar(widgets=widgets, maxval=len(label))
            pbar.start()

            for sub_label in os.listdir(os.path.join(data_repo, label)):
                images = os.listdir(os.path.join(data_repo, label, sub_label))

                ind = 0
                #for Xs, Ys in pool.map(self.preprocess_image, image_label_maps):
                Xs = []
                for img in images:
                    img = os.path.join(os.path.join(data_repo, label, sub_label,img))
                    x = self.preprocess_image(img)
                    Xs.append(x)

                pbar.update(ind)
                ind += 1
                X.append(Xs)
                y.append(ys)
            pbar.finish()

        X = np.array(X)
        y = np.array(y)
        print("final output", X.shape)
        #X = np.expand_dims(X, axis=5)
        #y = self.one_hot_encode(np.array(y))

        print(y.shape)
        return X, y