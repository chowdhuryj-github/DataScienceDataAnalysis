
# importing numpy
import numpy as np

# creating arrays from python lists
print(np.array([1, 4, 2, 5, 3]), "\n")

# numpy is constrained to arrays that all have the same type
print(np.array([3.14, 4., 2, 3]), "\n")

# set data type of the resulting array
print(np.array([1, 2, 3, 4], dtype="float"), "\n")

# initializing a multidimensional array
print(np.array([range(i, i+3) for i in [2, 4, 6]]), "\n")

# creating a 10 long integer array with just zeros
print(np.zeros(10, dtype=int), "\n")

# creating a 3x5 floating point array filled with ones
print(np.ones((3, 5), dtype="float"), "\n")

# create a 3x5 array filled with 3.14
print(np.full((3, 5), 3.14), "\n")

# create a array filled with linear sequence
print(np.arange(0, 20, 2), "\n")

# array of five values evenly spaced between 0 and 1
print(np.linspace(0, 1, 5), "\n")

# create a 3x3 array of uniformly distributed random values between 0 and 1
print(np.random.random((3, 3)), "\n")

# create a 3x3 array of normally distributed random values with mean 0 and SD 1
print(np.random.normal(0, 1, (3, 3)), "\n")

# create a 3x3 array of random integers in the interval [0, 10)
print(np.random.randint(0, 10, (3, 3)), "\n")

# create a 3x3 identity matrix
print(np.eye(3), "\n")

# create a uninitialized array of three integers 
print(np.empty(3))
