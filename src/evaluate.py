import os
import json
import numpy as np
from tensorflow import keras
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = np.load("data/processed/data.npz")
model = keras.models.load_model("models/model.h5")
loss, acc = model.evaluate(d["x_test"], d["y_test"], verbose=0)
pred = model.predict(d["x_test"]).argmax(axis=1)
os.makedirs("reports", exist_ok=True)
ConfusionMatrixDisplay(confusion_matrix(d["y_test"], pred)).plot()
plt.savefig("reports/confusion_matrix.png")
json.dump({"test_loss": float(loss), "test_accuracy": float(acc)}, open("metrics.json", "w"))
print("test_accuracy:", acc)
