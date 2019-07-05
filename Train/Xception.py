from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
from keras.applications import Xception
from keras.callbacks import ModelCheckpoint
from keras.models import Model
import numpy as np

mcp_save = ModelCheckpoint('./model.h5', save_best_only=True, monitor='val_loss', mode='min')

img_width, img_height = 128, 128
n_batch_size = 120
n_epochs = 1000

train_datagen = ImageDataGenerator(horizontal_flip=True,
                                    vertical_flip=True,
                                    validation_split=0.2,
                                    rescale=1./255.)

train_generator=train_datagen.flow_from_directory('./train', 
                                            target_size=(img_height, img_width), 
                                            batch_size=n_batch_size,
                                            shuffle=True,
                                            subset='training')

vali_generator = train_datagen.flow_from_directory('./train',
                                                  target_size=(img_height, img_width),
                                                  batch_size=n_batch_size,
                                                  shuffle=True,
                                                  subset='validation')

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

model.fit_generator(train_generator,
#                    steps_per_epoch = n_training_steps_per_epoch,
                    validation_data = vali_generator,
                    epochs = n_epochs,
#                    validation_steps = n_validation_steps_per_epoch,
                    class_weight='balanced',
                    callbacks=[mcp_save])

model.save('Xception_model.h5')

input('')
