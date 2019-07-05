from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.callbacks import ModelCheckpoint
from keras.models import Model
import numpy as np
import os
import json

mcp_save = ModelCheckpoint('./InceptionResNetV2.h5', save_best_only=True, save_weights_only=True, monitor='val_loss', mode='min')

img_width, img_height = 299, 299
n_batch_size = 30
n_epochs = 1
#n_training_steps_per_epoch = (2528//84)
#n_validation_steps_per_epoch = n_training_steps_per_epoch*0.2

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

# 以訓練好的 Xception 為基礎來建立模型
net = InceptionResNetV2(input_shape=(img_height, img_width, 3),
                              include_top=False, 
                              weights='imagenet', 
                              pooling='max')

# 增加 Dense layer 與 softmax 產生個類別的機率值
x = net.output
x = Dense(2048,  activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(128,  activation='relu')(x)
output_layer = Dense(6, activation='softmax', name='softmax')(x)

# 設定要進行訓練的網路層
model = Model(inputs=net.input, outputs=output_layer)

# 取ImageNet中的起始Weight，不使他隨機產生，故凍結最底層
FREEZE_LAYERS = 1
for layer in model.layers[:FREEZE_LAYERS]:
    layer.trainable = False
for layer in model.layers[FREEZE_LAYERS:]:
    layer.trainable = True

print('\n')
print('Trainable layers:')
for x in model.trainable_weights:
    print(x.name)

model.compile(optimizers.Adam(lr=1e-4), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])
print('\n')

# 輸出整個網路結構
model.summary()

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
                    json_file.write(model_json)

model.fit_generator(train_generator,
#                    steps_per_epoch = n_training_steps_per_epoch,
                    validation_data = vali_generator,
                    epochs = n_epochs,
#                    validation_steps = n_validation_steps_per_epoch,
                    class_weight='balanced',
                    callbacks=[mcp_save])


# serialize weights to HDF5
model.save_weights("./model_InceptionResNetV2.h5")
print("Saved model to disk")

##input('')
