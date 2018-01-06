import numpy as np
import os


def char_to_binary_float(n):
    return [float(ord(c) - 48) for c in "{0:b}".format(n)]


def int_array_to_binary_float(a):
    return [char_to_binary_float(it) for it in a]


input_size = 32
output_size = (input_size // 2) + 1
m = 10
train_size = int(m * 0.8)

np.random.seed(42)
a_numbers = np.random.randint(2 ** (input_size // 2), size=m)
b_numbers = np.random.randint(2 ** (input_size // 2), size=m)
y_numbers = [a_numbers[i] + b_numbers[i] for i in range(0, len(a_numbers))]
a = int_array_to_binary_float(a_numbers)
b = int_array_to_binary_float(b_numbers)
x = [a[i] + b[i] for i in range(0, len(a))]
y = int_array_to_binary_float(y_numbers)

x_train = x[:train_size]
y_train = y[:train_size]
x_test = x[train_size:m]
y_test = y[train_size:m]

print(np.shape(x), np.shape(x_train))
print(x)

dir_path = os.path.dirname(os.path.realpath(__file__))
np.save(dir_path + "/x-train.npy", x_train)
print(np.shape(x_train), np.shape(np.load(dir_path + "/x-train.npy")))
np.save(dir_path + "/y-train.npy", y_train)
np.save(dir_path + "/x-test.npy", x_test)
np.save(dir_path + "/y-test.npy", y_test)

print("done")