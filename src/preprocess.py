import os
import yaml
import numpy as np
from sklearn.model_selection import train_test_split


def normalize(a):
    a = a / 255.0  # scale to [0, 1]
    return (a - 0.286) / 0.353  # standardize


p = yaml.safe_load(open("params.yaml"))["preprocess"]
d = np.load("data/raw/fashion.npz")
x = normalize(d["x_train"])
x_tr, x_val, y_tr, y_val = train_test_split(
    x, d["y_train"], test_size=p["test_size"], random_state=p["seed"])
os.makedirs("data/processed", exist_ok=True)
np.savez("data/processed/data.npz", x_train=x_tr, y_train=y_tr,
         x_val=x_val, y_val=y_val,
         x_test=normalize(d["x_test"]), y_test=d["y_test"])
print("Saved processed data")
