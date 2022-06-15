import numpy as np

a = np.zeros((2,3,3,3))
a[0,0] = 0*np.eye(3)
a[0,1] = 1*np.eye(3)
a[0,1,0,2] = 1
a[0,2] = 2*np.eye(3)
a[0,2,0,2] = 2
a[1,0] = 10*np.eye(3)
a[1,1] = 11*np.eye(3)
a[1,2] = 12*np.eye(3)
a[1,2,0,2] = 22
#print(a)
b = a.transpose((0, 2, 1, 3)).reshape((6,9))
print(b)

c = np.zeros((3,3,3,3))
c[0, 0] = np.eye(3)
c[1, 1] = 2*np.eye(3)
c[2, 2] = 3*np.eye(3)
c2 = c[1:] - c[:-1]
print(c2.shape)
print(c2)
d = c2.transpose((0, 2, 1, 3)).reshape((6,9))
print(d)
e = np.outer(np.arange(3),np.ones(3))

f = e.reshape((9,))
print(f)
print(d @ f)
