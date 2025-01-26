
# importing numpy
import numpy as np

# seed for reporducibility
np.random.seed(0)

# generating a 1D, 2D and 3D view
x1 = np.random.randint(10, size=6)
x2 = np.random.randint(10, size=(3,4))
x3 = np.random.randint(10, size=(3, 4, 5))

# printing out the arrays
print("One Dimensional Array: ", x1, "\n")
print("Two Dimensional Array: ", "\n", x2, "\n")
print("Three Dimensional Array: ", "\n",  x3, "\n")

# each array has number of dimensions (ndim), shape (size) and size (total size)
# each array also has a data type, itemsize (size of each element) and nbytes (total size of array)
print("x3 ndim: ", x3.ndim)
print("x3 shape: ", x3.shape)
print("x3 size: ", x3.size)
print("dtype: ", x3.dtype)
print("itemsize: ", x3.itemsize, "bytes")
print("nbytes: ", x3.nbytes, "bytes")

# access single elements
print("One Dimensional Array: ", x1)
print("First Element: ", x1[0])
print("Fifth Element: ", x1[4])

# indexing from the end of the array
print("Last Index: ", x1[-1])
print("Second Last Index: ", x1[-2])

# multidimensional array
print("Two Dimensional Array: ", "\n",  x2)
print("The Zeroth Index: ", x2[0, 0])

# modifying the values 
x2[0,0] = 12
print("Modified 2D Array: ", x2)

# numpy arrays are a fixed type
x1[0] = 4.14159
print("Modified One Dimensional Array: ", x1)