import numpy as np
import os

input_size = 9
output_size = 19

x_train = np.random.random((1, input_size))
y_train = np.random.randint(output_size, size=(1, output_size))

x_test = np.random.random((1, input_size))
y_test = np.random.random((1, output_size))

dir_path = os.path.dirname(os.path.realpath(__file__))
np.savetxt(dir_path + "/x-train.txt", x_train)
np.savetxt(dir_path + "/y-train.txt", y_train)
np.savetxt(dir_path + "/x-test.txt", x_test)
np.savetxt(dir_path + "/y-test.txt", y_test)
