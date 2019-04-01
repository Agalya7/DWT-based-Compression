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

img_name = "img02"
org_img = cv2.imread(img_name + ".jpg", cv2.IMREAD_COLOR)
img = np.float32(cv2.imread(img_name + ".jpg", cv2.IMREAD_COLOR))/255

ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
y, cb, cr = cv2.split(ycbcr)
ycbcr_scaled = np.uint8(255.*(ycbcr - ycbcr.min())/(ycbcr.max() - ycbcr.min()))
#cv2.imshow('display', org_img)
#cv2.imshow('display', ycbcr_scaled)
#cv2.waitKey()
cv2.destroyAllWindows()

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
#newLL = zeros([len(org_img), len(org_img[0])])
#for i in range(len(LL)):
#    for j in range(len(LL[0])):
#        newLL[i][j*2] = LL[i][j]
#newHH = zeros([len(org_img), len(org_img[0])])
#for i in range(len(HH)):
#    for j in range(len(HH[0]), 2):
#        newHH[i][j*2] = HH[i][j]
newcb = zeros([len(LL), len(LL[0])])
for i in range(len(cb)):
    for j in range(len(cb[0]), 2):
        newcb[i][j] = cb[i][j]
newcr = zeros([len(LL), len(LL[0])])
for i in range(len(cr)):
    for j in range(len(cr[0]), 2):
        newcr[i][j] = cr[i][j]
print (len(LL), len(HH), len(newcb), len(newcr), len(org_img))
print (len(LL[0]), len(HH[0]), len(newcb[0]), len(newcr[0]), len(org_img[0]))

coeffs = (LL, (newcb, newcr, HH))
img = pywt.idwt2(coeffs, 'haar')
scipy.misc.imsave('textured.png', img)