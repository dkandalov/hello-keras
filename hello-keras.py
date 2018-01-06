from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical
import numpy as np
import os

input_size = 32
output_size = (input_size // 2) + 1
dir_path = os.path.dirname(os.path.realpath(__file__)) + "/datasets/addition/"

model = Sequential([
    Dense(32, activation='relu', input_shape=(input_size,)),
    Dense(output_size, activation='softmax')
])
print("layer sizes:")
for layer in model.layers:
    print(layer.input_shape, layer.output_shape)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# train
x_train = np.load(dir_path + "/x-train.npy")
y_train = np.load(dir_path + "/y-train.npy")
# one_hot_labels = to_categorical(y_train, num_classes=output_size + 1)
print(np.shape(x_train))
model.fit(x_train, y_train, epochs=10, batch_size=1)

# evaluate
x_test = np.load(dir_path + "/x-test.npy")
y_test = np.load(dir_path + "/y-test.npy")
score = model.evaluate(x_test, y_test, batch_size=128)
print("score", score)

