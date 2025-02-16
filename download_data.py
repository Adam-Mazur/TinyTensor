from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np
import os

print("Downloading MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
print("Downloaded MNIST dataset")

mnist.target = mnist.target.astype(np.uint8)
mnist.data = mnist.data.astype(np.uint8)

X, y = mnist.data[mnist.target <= 1], mnist.target[mnist.target <= 1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("The shape of the training and testing datasets are:")
print(X_train.shape, X_test.shape)

os.makedirs('data', exist_ok=True)

print("Saving the training and testing datasets to disk...")
with open("data/train.bin", "wb") as f:
    f.write(y_train.tobytes())  
    f.write(X_train.tobytes())

with open("data/test.bin", "wb") as f:
    f.write(y_test.tobytes())  
    f.write(X_test.tobytes())
print("Saved the training and testing datasets to disk")