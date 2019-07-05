import os
import cv2
from PIL import Image, ImageEnhance
import numpy as np
np.random.seed(1337)  # for reproducibility
##from keras.datasets import mnist
from keras.utils import np_utils, multi_gpu_model 
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.models import load_model
import time
import glob
from PIL import Image, ImageFont, ImageDraw
import csv


class Predict():
    def __init__(self):
        self.x = 350
        self.y = 300
        self.w = 512
        self.h = 512
        self.X_predict = []
        self.model = Sequential()
        self.result_list = []

        self.Load_model()
        self.check_folder()
        return
    
    def Load_model(self):
        self.model = load_model("./0409-1(0.9854500).h5")
##        self.model.load_weights('./weight.h5')
        self.model.summary()
        return

    def check_folder(self):
        default_dir = os.getcwd() + "/train"
        self.folders= os.listdir(default_dir)
        return

    def predict_data(self, check, image, image_512):
        if check == 1:
            self.Load_model()
            self.check_folder()
        self.X_predict.append(image_512)
        X_predict_np = np.asarray(self.X_predict)
        self.X_predict.clear()
        X_predict_np = X_predict_np.reshape(-1, 1, 128, 128)/255.
##        print(X_predict_np.shape)
        result = self.model.predict(X_predict_np, batch_size=None, verbose=0, steps=None)
        result_val = np.max(result)
        result = self.folders[int(result.argmax(axis=1))]
        self.result_list.append(result)

##        image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
##        cv2.putText(image_color, result, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)
##        cv2.imshow("Result", image_color)
##        cv2.waitKey(1000)
        return

    def write_csv(self):
        with open('output.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['ID', 'Label'])
            for i, j  in enumerate(self.result_list):
                if int(j) == 0:
                    writer.writerow(["test_" + str(i).zfill(5) + ".png", "0"])
                elif int(j) == 1:
                    writer.writerow(["test_" + str(i).zfill(5) + ".png", "1"])
                elif int(j) == 2:
                    writer.writerow(["test_" + str(i).zfill(5) + ".png", "2"])
                elif int(j) == 3:
                    writer.writerow(["test_" + str(i).zfill(5) + ".png", "3"])
                elif int(j) == 4:
                    writer.writerow(["test_" + str(i).zfill(5) + ".png", "4"])
                elif int(j) == 5:
                    writer.writerow(["test_" + str(i).zfill(5) + ".png", "5"])
        return

n = Predict()
##files = glob.glob('./test_images/*.png')
##test_00000.png
##test_10141.png

for i in range(10142):
    img = Image.open("./test_images/test_" + str(i).zfill(5) + ".png")
    img_original = np.asarray(img)
    img = cv2.resize(img_original, (128, 128), interpolation=cv2.INTER_CUBIC)
    n.predict_data(0,img_original, img)
n.write_csv()
