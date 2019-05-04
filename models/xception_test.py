from keras.models import Model
from keras.layers import Input, Conv2D, Activation, Dropout, Lambda, Dense, BatchNormalization
from keras.layers import MaxPooling2D, AveragePooling2D, concatenate, Add
from keras.layers import GlobalMaxPooling2D, GlobalAveragePooling2D, merge, Flatten
from keras.applications import Xception

in_layer = Input((280, 200, 3))
        
# use keras existing resnet50
xceptionModel = Xception(include_top=False, weights='imagenet', 
input_tensor=Input(shape=(280, 200, 3)), pooling="avg")

kernel = Dropout(0.5) (xceptionModel (in_layer))
xceptionModel.summary()
print(kernel.shape)

denlayer = kernel
# denlayer = GlobalMaxPooling2D() (kernel)
# denlayer = Flatten() (kernel)

# adding dense layers
denlayer = Dropout(0.5) (Dense(100) (denlayer))

out_layer = Dense(1, activation='linear') (denlayer)
model = Model(inputs=[in_layer], outputs=[out_layer])

model.summary()