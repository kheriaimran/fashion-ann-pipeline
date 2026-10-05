import os
import numpy as np
from tensorflow import keras

(x_tr, y_tr), (x_te, y_te) = keras.datasets.fashion_mnist.load_data()
os.makedirs("data/raw", exist_ok=True)
np.savez("data/raw/fashion.npz", x_train=x_tr, y_train=y_tr, x_test=x_te, y_test=y_te)
print("Saved raw data")
