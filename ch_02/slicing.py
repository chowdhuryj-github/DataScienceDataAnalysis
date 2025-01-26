
# importing numpy
import numpy as np

# slicing sub arrays x[start:stop:step]
x = np.arange(10)
print("Array: ", x)

# slicing the first five elements
print("First five elements: ", x[:5])

# elements after index 5
print("Elements after 5th Index: ", x[5:])

# the middle subarray
print("Middle Subarray: ", x[4:7])

# every other element
print("Every other element: ", x[::2])

# every other element starting from index 1
print("Starting from 1st index:", x[1::2])