import numpy as np

a = np.array([1,2,3])
b = np.array([4,5])

print(a.shape)
a = a.reshape(3,1)

print(a.shape)
print(b.shape)

c = a*b
print(c)
print(c.shape)