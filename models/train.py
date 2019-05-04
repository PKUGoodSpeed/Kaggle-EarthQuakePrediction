"""
Generic trainer class for RNN and CNN
@Author: pkugoodspeed
"""

import os
import gc
import sys
import numpy as np
import pandas as pd
from datetime import datetime

ROOT = "/home/zebo/Workspace/kaggle/Kaggle-EarthQuakePrediction"
H5_DATA_PATH = os.path.join(ROOT, "input", "train_chunk_h5")
sys.path.append(ROOT)

from utils.h5io import load_from_h5


GLR = 0.01 # Global Learning Rate
GDR = 0.92 # Global decaying rate
DC_EPOCH = 2 # Number of epochs for decaying


class NNModelTrainer:
    
    # Split raw training data into chunks
    _num_chunks = 5
    _chunk_index = 0
    
    # Load data row by row in to generator
    _row_index = None
    _data = None
    _data_length = None
    _load_data = None
    _block_size = None
    _overlap = None

    # mini batch
    _batch_size = None

    # FFT coeff
    _fs = 100000
    _fft_win = 50
    _nperseg = 550
    _fft_ol = 50
    _width = 280
    _height = 200
    
    ## About model
    
    def __init__(self, block_size=100000, overlap=1000):
        self._row_index = 0
        self._load_data = True
        self._block_size = block_size
        self._overlap = overlap
        self._fs = block_size

    def getFilepath(self):
        return os.path.join(H5_DATA_PATH, "chunk{}.h5".format(str(self._chunk_index)))

    def loadChunkData(self):
        assert self._load_data, "Not a right time to load chunk data."
        self._data = load_from_h5(self.getFilepath())
        self._data.columns = ["strain", "time"]
        self._data = self._data.astype(float)
        self._row_index = 0
        self._data_length = self._data.shape[0]
        self._chunk_index = (self._chunk_index + 1) % self._num_chunks

    def processData(self, raw_input):
        freq, times, spec = signal.spectrogram(
            raw_input, fs=self._fs, window=("kaiser", self._fft_win), nperseg=self._nperseg, noverlap=self._ffs_ol)
        p1 = max(0, NR - np.shape(spec)[0])
        p2 = max(0, NC - np.shape(spec)[1])
        spec = np.pad(spec, [(0, p1), (0, p2)], mode='constant')
        return spec

    def trainGenerator(self):
        X = []
        Y = []
        while True:
            if self._load_data:
                del self._data
                gc.collect()
                self.loadChunkData()
            
            while self._row_index <= self._data_length - self._block_size:
                img_ = self.process_data(
                    np.array(
                        self._data[self._row_index: self._row_index + self._block_length]["strain"].tolist()
                    )
                )
                print(img_.shape)
                tar_ = self._data["time"].loc[self._row_index + self._block_length - 1]
                X.append(img_)
                Y.append(tar_)
                self._row_index += self._overlap

                if len(X) == self._batch_size:
                    if len(Y) == self._batch_size:
                        yield (np.array(X), np.array(Y))
                    del X
                    del Y
                    X = []
                    Y = []
                    gc.collect()
            if len(X) > 0:
                if len(Y) == len(X):
                    yield (np.array(X), np.array(Y))
                del X
                del Y
                X = []
                Y = []
                gc.collect()
            self._load_data = True
    
    def loadModel(self, model):
        self._model = model
    
    def loadWeightsFromFile(model_file):
        print("Loading model from {FILE} ...".format(FILE=model_file))
        self._model.load_weights(model_file)
    
    def getModel(self):
        return self._model

    
    def train(self, class_weights=None, learning_rate=0.02, decaying_rate=0.9, epochs=10, resume=False):
        '''train the model'''
        # compile the model first
        self._model.compile(optimizer=Adam(0.005), loss="mean_absolute_error", metrics="mean_absolute_error")
        checker_path = os.path.join(ROOT, self._model.get_name())

        if not os.path.exists(checker_path):
            os.makedirs(checker_path)
        checker_file = os.path.join(
            checker_path, str(datetime.now()).replace(" ", "-") + ".h5")

        global GLR
        global GDR
        ## Setting learning rate explicitly
        GLR = learning_rate
        GDR = decaying_rate
        
        ## Adaptive learning rate changing
        def scheduler(epoch):
            global GLR
            global GDR
            if epoch % DC_EPOCH == 0:
                GLR *= GDR
                print("CURRENT LEARNING RATE = " + str(GLR))
            return GLR

        change_lr = LearningRateScheduler(scheduler)
        
        ## Set early stopper:
        earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')
        
        ## Set Check point
        checkpointer = ModelCheckpoint(filepath=checker_file, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

        train_steps = int((len(self._data)+self._batch_size-1)/self._batch_size)
        valid_steps = int((len(self._data)+self._batch_size-1)/self._batch_size)

        history = self.model.fit_generator(self.train_generator(), steps_per_epoch=train_steps, epochs=epochs, verbose=1, 
        callbacks=[earlystopper, checkpointer, change_lr], validation_data=self._valid_gen(), validation_steps=valid_steps,
        class_weight=class_weights)
        return history

def trainerTest():
    trainer = NNModelTrainer()
    trainer.trainGenerator()


if __name__ == "__main__":
    trainerTest()
    



    