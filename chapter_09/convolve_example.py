#
#  Illustrate how different NumPy and SciPy convolution
#  and correlation routines work.
#
import numpy as np
from scipy.signal import convolve2d
from scipy.misc import face
from PIL import Image

#  Get Ricky's face
img = face(True)
img = img[:512,(img.shape[1]-612):(img.shape[1]-100)]

#  An asymmetric kernel
k = np.array([[1,0,0],[0,-8,0],[0,0,3]])
c = convolve2d(img, k, mode='same')

#  Results
print("Original:")
print(img[:8,:8])
print()
print("Kernel:")
print(k)
print()
print("convolve2d(img,k,mode='same'):")
print(c[1:8,1:8])
print()

if (c.min() < 0):
    c = c + np.abs(c.min())
c = (255*(c / c.max())).astype("uint8")

Image.fromarray(c).save("ricky_convol.png")
Image.fromarray(img).save("ricky_orig.png")

