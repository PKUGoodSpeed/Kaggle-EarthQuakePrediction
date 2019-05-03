"""
@Deprecated
"""
import os
import gc
import numpy as np
import pandas as pd
from h5io import write_to_h5

DATA_PATH = '/home/goodspeed/Documents/workspace/Kaggle/Kaggle-EarthQuakePrediction/input'
NUM_ROWS = 629145480
CHUNK_SIZE = 10000000
OVERLAP = 1000
TRAIN_PATH = os.path.join(DATA_PATH, "train")
TRAIN_H5_PATH = os.path.join(DATA_PATH, "train_h5")


def split(chunksize, overlap):
    if not os.path.exists(TRAIN_PATH):
        os.makedirs(TRAIN_PATH)
    if not os.path.exists(TRAIN_H5_PATH):
        os.makedirs(TRAIN_H5_PATH)
    raw_train_file = os.path.join(DATA_PATH, "train.csv")
    for i, j in enumerate(range(0, NUM_ROWS, chunksize)):
        df = pd.read_csv(
            raw_train_file,
            dtype={
                'acoustic_data': np.int16,
                'time_to_failure': np.float32
            },
            skiprows=j,
            nrows=min(chunksize + overlap, NUM_ROWS - j)
        )
        csv_path = os.path.join(TRAIN_PATH, "chunk" + str(int(i)) + ".csv")
        h5_path = os.path.join(TRAIN_H5_PATH, "chunk" + str(int(i)) + ".h5")
        # df.to_csv(csv_path, index=False)
        write_to_h5(filename=h5_path, df=df)
        print(i, j, df[: 1])
        del df
        gc.collect()
    print("DONE!")


if __name__ == "__main__":
    split(chunksize=CHUNK_SIZE, overlap=OVERLAP)
