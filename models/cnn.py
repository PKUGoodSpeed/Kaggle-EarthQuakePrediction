from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Flatten, Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop, Adadelta

IMG_HEIGHT = 280
IMG_WIDTH = 200


def getSimpleCNNModel():
    ### Construct the model
    print("CONSTRUCTING MODEL!")
    model = Sequential()
    model.add(MaxPooling2D(pool_size = (2, 2), input_shape = (IMG_HEIGHT, IMG_WIDTH, 1)))
    
    model.add(Conv2D(64, kernel_size = (7, 7), padding = 'same'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(LeakyReLU(alpha=0.01))
    #model.add(BatchNormalization())
    model.add(Dropout(hyper_dropout1))
    
    model.add(Conv2D(128, kernel_size = (5, 5), padding = 'same'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(LeakyReLU(alpha=0.01))
    #model.add(BatchNormalization())
    model.add(Dropout(hyper_dropout2))
    
    model.add(Conv2D(256, kernel_size = (3, 3), padding = 'same'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(LeakyReLU(alpha=0.01))
    #model.add(BatchNormalization())
    model.add(Dropout(hyper_dropout3))


    model.add(Flatten())
    
    model.add(Dense(1024))
    model.add(Activation('sigmoid'))
    model.add(Dropout(hyper_dropout5))
    
    model.add(Dense(n_cls, activation = 'softmax'))
    model.summary()