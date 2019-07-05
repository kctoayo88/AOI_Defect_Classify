import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import gc
from PIL import Image, ImageEnhance
from keras.utils import np_utils, multi_gpu_model
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, Conv2D, AveragePooling2D
from keras.optimizers import SGD
from keras.optimizers import Adam, Adadelta
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.utils.multiclass import type_of_target

class Data_setup():
    def __init__(self, path, size):
        self.set_data_x = []
        self.set_data_y = []
        self.X_train = []
        self.y_train = []
        self.len_list = []
        self.path = path
        self.size = size
        return

    def Process_data(self):
        X_train, y_train = self.Data_balance()
        print((np.asarray(X_train)).shape, len(y_train))
        X_train, X_test, y_train, y_test = self.Rescale_shuffle(X_train, y_train)
        return X_train, X_test, y_train, y_test

    def Clear_data(self):
        self.X_train.clear()
        self.y_train.clear()
        return

    def Rescale_shuffle(self, X_train, y_train):
        max_cl = np.max(y_train)
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        X_train = X_train.reshape(-1, 1, self.size, self.size)/255.
        y_train = np_utils.to_categorical(y_train, num_classes=6)
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, train_size = 0.7, test_size=0.3, random_state=42)
        return X_train, X_test, y_train, y_test

    def Data_balance(self):
        folders= os.listdir(self.path)
        for i, folder_name in enumerate(folders):
            new_path  = self.path + "/" + str(folder_name)
            files_= os.listdir(new_path)
            self.len_list.append(len(files_))
        max_class = np.max(self.len_list)

        for i, folder_name in enumerate(folders):
            new_path  = self.path + "/" + str(folder_name)
            files_= os.listdir(new_path)
##              copy_num = max_class - len(files_)
            times = int(max_class/len(files_))
            remainder = max_class%len(files_)
            for j in files_:
                image = Image.open(new_path + "/" + str(j))
                image_data = np.asarray(image)
                image_data = cv2.resize(image_data, (self.size, self.size), interpolation=cv2.INTER_CUBIC)
                self.set_data_x.append(image_data)
                self.set_data_y.append(i)
            for z in range(times):
                self.X_train[len(self.X_train):len(self.X_train) + len(files_)] = self.set_data_x[:]
                self.y_train[len(self.y_train):len(self.y_train) + len(files_)] = self.set_data_y[:]
            self.X_train[len(self.X_train):len(self.X_train) + remainder] = self.set_data_x[:remainder]
            self.y_train[len(self.y_train):len(self.y_train) + remainder] = self.set_data_y[:remainder]
            self.set_data_x.clear()
            self.set_data_y.clear()
            X_train = self.X_train
            y_train = self.y_train
        for i in range(2):
            X_train = X_train + self.X_train
            y_train = y_train + self.y_train
##        X_train, y_train = self.X_train, self.y_train
        return X_train, y_train

class CNN_model():
    def __init__(self, X_train, X_test, y_train, y_test, size):
        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test
        self.size = size
        self.set_model()
        return

    def set_model(self):
        self.model = Sequential()
        self.model.add(Convolution2D(
            batch_input_shape=(None, 1, self.size, self.size),
            filters=128,
            kernel_size=7,
            strides=1,
            padding='same',     # Padding method
            data_format='channels_first',
        ))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(
            pool_size=2,
            strides=2,
            padding='same',    # Padding method
            data_format='channels_first',
        ))

        self.model.add(Convolution2D(256, 5, strides=1, padding='same', data_format='channels_first'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(2, 2, 'same', data_format='channels_first'))

        self.model.add(Convolution2D(512, 5, strides=1, padding='same', data_format='channels_first'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(2, 2, 'same', data_format='channels_first'))

        self.model.add(Convolution2D(1024, 3, strides=1, padding='same', data_format='channels_first'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(2, 2, 'same', data_format='channels_first'))

        self.model.add(Convolution2D(1024, 3, strides=1, padding='same', data_format='channels_first'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(2, 2, 'same', data_format='channels_first'))

        self.model.add(Flatten())

        self.model.add(Dense(1024))
        self.model.add(Dropout(0.5))
        self.model.add(Activation('relu'))

        self.model.add(Dropout(0.5))

        self.model.add(Dense(128))
        self.model.add(Dropout(0.5))
        self.model.add(Activation('relu'))

        self.model.add(Dense(6))
        self.model.add(Activation('softmax'))
        adam = Adam(lr=1e-4)
        
        self.model.summary()

        self.model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
        return

    def train(self, num, title):
        mcp_save = ModelCheckpoint('./Best_model.h5', save_best_only=True, monitor='val_loss', mode='min')
        print('Training ------------')
        self.keras_cnn = self.model.fit(self.X_train, self.y_train, epochs=50000, batch_size=100,
                                        verbose=1, validation_data = (self.X_test, self.y_test),
                                        callbacks=[mcp_save])##120##
        self.model.save("./Best_model.h5")
        self.Save_info()##Save loss and accuracy
        self.plot_(num, title)
        return

def main():
    default_dir = os.getcwd() + "/train"
    num = 0
    data_setup_ba = Data_setup(default_dir, 128)
    X_train, X_test, y_train, y_test = data_setup_ba.Process_data()

    cnn_cl_ba = CNN_model(X_train, X_test, y_train, y_test, 128)
##    cnn_cl_ba.train(num, name_list[num])
    cnn_cl_ba.train(1, "CNN -Balance Data")
    
if __name__ == '__main__':
    main()
