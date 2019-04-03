# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 22:45:29 2019

@author: 15pt03
"""

import matplotlib.pyplot as plt
import cv2
import math
import os
import numpy as np
from numpy import zeros
from numpy.linalg import norm
import scipy.misc
import pywt
import pywt.data

def LtoRShift(block):
    return np.transpose(block)

def RtoLShift(block):
    new = np.zeros(block.shape)
    for i in range(len(block)-1, -1, -1):
        for j in range(len(block)-1-i):
            new[i][j] = block[len(block)-1-j][len(block)-1-i]
    for i in range(len(block)-1, -1, -1):
        new[i][len(block)-1-i] = block[i][len(block)-1-i]
    for i in range(len(block)-1, -1, -1):
        for j in range(len(block)-i, len(block)):
            new[i][j] = block[len(block)-1-j][len(block)-1-i]
    return new

img_name = "img04"
org_img = cv2.imread(img_name + ".jpg", cv2.IMREAD_COLOR)
img = np.float32(cv2.imread(img_name + ".jpg", cv2.IMREAD_COLOR))/255

ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
y, cb, cr = cv2.split(ycbcr)
ycbcr_scaled = np.uint8(255.*(ycbcr - ycbcr.min())/(ycbcr.max() - ycbcr.min()))

titles = ['img1', ' img2']
cfs = pywt.dwt(y, 'haar')
CA, CB = cfs
fig = plt.figure(figsize=(3, 3))
for i, a in enumerate([CA, CB]):
    ax = fig.add_subplot(1, 2, i + 1)
    ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
    ax.set_title(titles[i], fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
fig.tight_layout()
plt.show()

i = pywt.idwt(cb, cr, 'haar')
plt.imshow(i)

titles = ['img1', ' img2', 'img3', 'img4']
coeffs2 = pywt.dwt2(y, 'haar')
LL, (LH, HL, HH) = coeffs2
fig = plt.figure(figsize=(12, 3))
for i, a in enumerate([LL, LH, HL, HH]):
    ax = fig.add_subplot(2, 2, i + 1)
    ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
    ax.set_title(titles[i], fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
fig.tight_layout()
plt.show()

newcb = zeros([len(LL), len(LL[0])])
for i in range(len(cb)):
    for j in range(len(cb[0]), 2):
        newcb[i][j] = cb[i][j]
newcr = zeros([len(LL), len(LL[0])])
for i in range(len(cr)):
    for j in range(len(cr[0]), 2):
        newcr[i][j] = cr[i][j]

coeffs = (LL, (newcb, newcr, HH))
img = pywt.idwt2(coeffs, 'haar')
incr = 30
factor = min(img.shape) % incr
factor = min(img.shape) - factor
img = img[:factor, :factor]
scipy.misc.imsave('textured.png', img)
print (img.shape)
block = []
for i in range(0, factor, incr):
    for j in range(0, factor, incr):
        block.append(img[i:i+incr, j:j+incr])
print (len(block), len(block[0]), len(block[0][0]))
for i in range(0, len(block), 2):   #even
    block[i] = LtoRShift(block[i])
for i in range(1, len(block), 2):   #odd
    block[i] = RtoLShift(block[i])
block_incr = len(img)//incr
new_image = np.block([block[0:block_incr]])
for i in range(1, block_incr):
    b = np.block([block[i*block_incr:(i+1)*block_incr]])
    print (i, new_image.shape, b.shape)
    new_image = np.vstack([new_image, b])
print (new_image.shape)
scipy.misc.imsave('tringular.png', new_image)
for i in range(0, len(block), 2):   #even
    block[i] = LtoRShift(block[i])
for i in range(1, len(block), 2):   #odd
    block[i] = RtoLShift(block[i])
new_image = np.block([block[0:block_incr]])
for i in range(1, block_incr):
    b = np.block([block[i*block_incr:(i+1)*block_incr]])
    print (i, new_image.shape, b.shape)
    new_image = np.vstack([new_image, b])
scipy.misc.imsave('rev-tringular.png', new_image)
