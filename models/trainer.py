"""
Generic trainer class for RNN and CNN
@Author: pkugoodspeed
"""

import os
import gc
import sys
import time
import numpy as np
import pandas as pd
from scipy import signal
from scipy.fftpack import fft
from datetime import datetime
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler, Callback, EarlyStopping, ModelCheckpoint

ROOT = "/home/zebo/Workspace/kaggle/Kaggle-EarthQuakePrediction"
sys.path.append(ROOT)                       # Add root directory to the import path

from utils.h5io import load_from_h5

GLR = 0.01                                  # Global Learning Rate
GDR = 0.92                                  # Global decaying rate
EPD = 2                                     # Epochs per decaying


class NNModelTrainer:
    """ Provide util methods for NN usage """
    # loading data via chunks to void memory explosion
    NUM_TRAIN_CHUNKS = 5
    NUM_VALID_CHUNKS = 2
    NUM_TRAIN_STEPS = 2815
    NUM_VALID_STEPS = 1120

    # Block parameters that are processed with fft
    BLOCK_SIZE = 100000
    BLOCK_STRIDE = 5000
    
    # FFT parameters
    FFT_WINDOW = 5
    FFT_NPERSEG = 566
    FFT_OVERLAP = 40
    FFT_DELTA = 1.E-8
    FFT_CONVERTED_HEIGHT = 284
    FFT_CONVERTED_WIDTH = 190
    
    # Image size
    IMG_HEIGHT = 288
    IMG_WIDTH = 192
    
    BATCH_SIZE = 32
    
    # Model related:
    _num_channel = 1
    _model = None
    _model_name = "Default_CNN"
    
    def __init__(self, num_channel=1):
        self._num_channel = num_channel
    
    def loadChunkTrain(self, chunk_index=0):
        assert chunk_index < self.NUM_TRAIN_CHUNKS, "Invalid train chunk index " + str(chunk_index)
        data_path = os.path.join(
            ROOT, "input", "train_chunk_h5", "train{IDX}.h5".format(IDX=str(chunk_index)))
        df = load_from_h5(data_path)
        df = df.astype(float)
        df.columns = ["strain", "time"]
        return df
    
    def loadChunkValid(self, chunk_index=0):
        assert chunk_index < self.NUM_VALID_CHUNKS, "Invalid valid chunk index " + str(chunk_index)
        data_path = os.path.join(
            ROOT, "input", "train_chunk_h5", "valid{IDX}.h5".format(IDX=str(chunk_index)))
        df = load_from_h5(data_path)
        df = df.astype(float)
        df.columns = ["strain", "time"]
        return df
    
    def loadDataTest(self, seg_id="seg_004cd2"):
        data_path = os.path.join(
            ROOT, "input", "test", "{SEG}.csv".format(SEG=str(seg_id)))
        df = pd.read_csv(data_path)
        df = df.astype(float)
        df.columns = ["strain"]

    def getSpectrum(self, raw_input):
        freq, times, spec = signal.spectrogram(
            raw_input,
            fs=self.BLOCK_SIZE,
            window=("kaiser", self.FFT_WINDOW),
            nperseg=self.FFT_NPERSEG,
            noverlap=self.FFT_OVERLAP)
        del freq
        del times
        gc.collect()
        assert spec.shape == (
            self.FFT_CONVERTED_HEIGHT,
            self.FFT_CONVERTED_WIDTH), "Shape error, need fix!"
        p1 = max(0, self.IMG_HEIGHT - spec.shape[0])
        p2 = max(0, self.IMG_WIDTH - spec.shape[1])
        spec = np.pad(spec, [(0, p1), (0, p2)], mode='constant')
        spec = np.log(np.array(spec) + self.FFT_DELTA)
        return spec
    
    
    def _dataGenerator(self, sample_type="train"):
        sample_type = sample_type.lower()
        assert sample_type in ["train", "valid"], "Invalid sample type: " + str(sample_type)

        # Using sample_type to distinguish between train and valid generator
        if sample_type == "train":
            num_chunks = self.NUM_TRAIN_CHUNKS
        else:
            num_chunks = self.NUM_VALID_CHUNKS

        X = []
        Y = []
        
        chunk_index = 0
        while True:
            if sample_type == "train":
                df = self.loadChunkTrain(chunk_index=chunk_index)
            else:
                df = self.loadChunkValid(chunk_index=chunk_index)
            print("Getting {STYPE} dataframe from chunk #{IDX} with shape: {SHAPE}".format(
                STYPE=str(sample_type), IDX=str(chunk_index), SHAPE=str(df.shape)))

            # Start processing data via block
            df_length = df.shape[0]
            row_index = 0
            while row_index <= df_length - self.BLOCK_SIZE:
                img_ = self.getSpectrum(
                    df["strain"][row_index: row_index + self.BLOCK_SIZE])
                img_ = np.stack(self._num_channel * [img_], axis=-1)
                assert img_.shape == (self.IMG_HEIGHT, self.IMG_WIDTH, self._num_channel)
                tar_ = df["time"].loc[row_index + self.BLOCK_SIZE - 1]
                X.append(img_)
                Y.append(tar_)
                row_index += self.BLOCK_STRIDE
                
                # Yield batches
                if len(X) == self.BATCH_SIZE:
                    if len(Y) == self.BATCH_SIZE:
                        yield (np.array(X), np.array(Y))
                    del X
                    del Y
                    X = []
                    Y = []
                    gc.collect()
            
            # Boundary Treatment
            if len(X) > 0:
                if len(Y) == len(X):
                    yield (np.array(X), np.array(Y))
                del X
                del Y
                X = []
                Y = []
                gc.collect()
            
            # Switch Chunk
            del df
            gc.collect()
            chunk_index = (chunk_index + 1) % num_chunks

    def fitGeneratorTrain(self):
        self._dataGenerator(sample_type="train")

    def fitGeneratorValid(self):
        self._dataGenerator(sample_type="valid")


    def loadModel(self, model):
        self._model = model
    
    def loadWeightsFromFile(self, model_file=None):
        if model_file is None:
            model_file = self.getModelName() + ".h5"
        model_file = os.path.join(ROOT, "nnmodels", model_file)
        assert os.path.exists(model_file)
        print("Loading model from {FILE} ...".format(FILE=model_file))
        self._model.load_weights(model_file)
    
    def getModel(self):
        return self._model
    
    def setModelName(self, model_name):
        self._model_name = model_name
    
    def getModelName(self):
        return self._model_name

    
    def train(
        self, learning_rate=0.02, decaying_rate=0.9,
        epochs_per_decay=2, epochs=10):
        '''train the model'''
        # compile the model first
        self._model.compile(
            optimizer=Adam(0.005),
            loss="mean_absolute_error",
            metrics=["mean_absolute_error"]
        )
        
        # Set checker points file
        checker_path = os.path.join(ROOT, "nnmodels")
        if not os.path.exists(checker_path):
            os.makedirs(checker_path)
        checker_file = os.path.join(checker_path, self.getModelName() + ".h5")

        global GLR
        global GDR
        global EPD
        ## Setting learning rate explicitly
        GLR = learning_rate
        GDR = decaying_rate
        EPD = epochs_per_decay
        
        ## Adaptive learning rate changing
        def scheduler(epoch):
            global GLR
            global GDR
            global EPD
            if epoch % EPD == 0:
                GLR *= GDR
                print("CURRENT LEARNING RATE = " + str(GLR))
            return GLR

        change_lr = LearningRateScheduler(scheduler)
        
        ## Set early stopper:
        earlystopper = EarlyStopping(monitor='val_loss', patience=7, verbose=1, mode='auto')
        
        ## Set Check point
        checkpointer = ModelCheckpoint(filepath=checker_file, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

        # Need manually setted when self.BLOCK_STRIDE is modified.
        train_steps = self.NUM_TRAIN_STEPS
        valid_steps = self.NUM_VALID_STEPS

        history = self._model.fit_generator(
            self.fitGeneratorTrain(),
            steps_per_epoch=train_steps,
            epochs=epochs,
            verbose=1, 
            callbacks=[earlystopper, checkpointer, change_lr],
            validation_data=self.fitGeneratorValid(),
            validation_steps=valid_steps)
        return history



class NNModelTrainerTest:
    
    _trainer = None

    def __init__(self):
        self._trainer = NNModelTrainer()
    
    def testLoadChunks(self):
        t_start = time.time()
        print("\n== Testing {NAME} ==".format(NAME=sys._getframe().f_code.co_name))
        # test loading train chunks
        for i in range(self._trainer.NUM_TRAIN_CHUNKS):
            df = self._trainer.loadChunkTrain(chunk_index=i)
            print("Train Chunk {IDX} => Columns: {COL}; Data Shape: {SHAPE}".format(
                IDX=str(i), COL=str(df.columns), SHAPE=str(df.shape)))
            del df
            gc.collect()

        # test loading valid chunks
        for i in range(self._trainer.NUM_VALID_CHUNKS):
            df = self._trainer.loadChunkValid(chunk_index=i)
            print("Valid Chunk {IDX} => Columns: {COL}; Data Shape: {SHAPE}".format(
                IDX=str(i), COL=str(df.columns), SHAPE=str(df.shape)))
            del df
            gc.collect()
        print("-------------")
        print("Time Usage for {NAME}: {TIME} sec\n\n".format(
            NAME=sys._getframe().f_code.co_name,
            TIME=str(time.time() - t_start)))
    
    def testGetSpectrum(self):
        t_start = time.time()
        print("\n== Testing {NAME} ==".format(NAME=sys._getframe().f_code.co_name))

        for train_chunk_index in range(self._trainer.NUM_TRAIN_CHUNKS):
            df = self._trainer.loadChunkTrain(chunk_index=train_chunk_index)
            df_length = df.shape[0]
            row_index = 0
            while row_index <= df_length - self._trainer.BLOCK_SIZE:
                spec = self._trainer.getSpectrum(
                    df["strain"][
                        row_index: row_index + self._trainer.BLOCK_SIZE])
                assert spec.shape == (self._trainer.IMG_HEIGHT, self._trainer.IMG_WIDTH)
                del spec
                gc.collect()
                row_index += 1000000
            print(row_index)
            del df
            gc.collect()
            
        for valid_chunk_index in range(self._trainer.NUM_VALID_CHUNKS):
            df = self._trainer.loadChunkValid(chunk_index=valid_chunk_index)
            df_length = df.shape[0]
            row_index = 0
            while row_index <= df_length - self._trainer.BLOCK_SIZE:
                spec = self._trainer.getSpectrum(
                    df["strain"][
                        row_index: row_index + self._trainer.BLOCK_SIZE])
                assert spec.shape == (self._trainer.IMG_HEIGHT, self._trainer.IMG_WIDTH)
                del spec
                gc.collect()
                row_index += 1000000
            print(row_index)
            del df
            gc.collect()
        
        print("-------------")
        print("Time Usage for {NAME}: {TIME} sec\n\n".format(
            NAME=sys._getframe().f_code.co_name,
            TIME=str(time.time() - t_start)))
    
    def testFitGenerator(self):
        t_start = time.time()
        print("\n== Testing {NAME} ==".format(NAME=sys._getframe().f_code.co_name))
        self._trainer.fitGeneratorTrain()
        self._trainer.fitGeneratorValid()
        
        print("-------------")
        print("Time Usage for {NAME}: {TIME} sec\n\n".format(
            NAME=sys._getframe().f_code.co_name,
            TIME=str(time.time() - t_start)))
    
    def testTemplate(self):
        t_start = time.time()
        print("\n== Testing {NAME} ==".format(NAME=sys._getframe().f_code.co_name))
        
        
        print("-------------")
        print("Time Usage for {NAME}: {TIME} sec\n\n".format(
            NAME=sys._getframe().f_code.co_name,
            TIME=str(time.time() - t_start)))
    
    def getNumSteps(self):
        t_start = time.time()
        print("\n== Testing {NAME} ==".format(NAME=sys._getframe().f_code.co_name))
        
        num_train_steps = 0
        num_valid_steps = 0

        for train_chunk_index in range(self._trainer.NUM_TRAIN_CHUNKS):
            df = self._trainer.loadChunkTrain(chunk_index=train_chunk_index)
            df_length = df.shape[0]
            row_index = 0
            n_spec = 0
            while row_index <= df_length - self._trainer.BLOCK_SIZE:
                n_spec += 1
                row_index += self._trainer.BLOCK_STRIDE
            del df
            gc.collect()
            num_train_steps += int((n_spec + self._trainer.BATCH_SIZE - 1) / self._trainer.BATCH_SIZE)
            
        for valid_chunk_index in range(self._trainer.NUM_VALID_CHUNKS):
            df = self._trainer.loadChunkValid(chunk_index=valid_chunk_index)
            df_length = df.shape[0]
            row_index = 0
            n_spec = 0
            while row_index <= df_length - self._trainer.BLOCK_SIZE:
                n_spec += 1
                row_index += self._trainer.BLOCK_STRIDE
            del df
            gc.collect()
            num_valid_steps += int((n_spec + self._trainer.BATCH_SIZE - 1) / self._trainer.BATCH_SIZE)
        print("========== Get number of training epoch steps: " + str(num_train_steps))
        print("========== Get number of validing epoch steps: " + str(num_valid_steps))
        
        print("-------------")
        print("Time Usage for {NAME}: {TIME} sec\n\n".format(
            NAME=sys._getframe().f_code.co_name,
            TIME=str(time.time() - t_start)))
        

    def testRun(self):
        self.testLoadChunks()
        self.testGetSpectrum()
        self.testFitGenerator()


if __name__ == "__main__":
    trainer_test = NNModelTrainerTest()
    trainer_test.testRun()
    trainer_test.getNumSteps()