# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 22:45:29 2019

@author: 15pt03
"""

import matplotlib.pyplot as plt
from numpy import zeros
import numpy as np
import scipy.misc
import pywt.data
import pywt
import cv2
import random


"""
-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
General Functions:
-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
"""
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

#stack together the blocks to create the image
def ConstructImage(block):
    new_image = np.block([block[0:block_incr]])
    for i in range(1, block_incr):
        b = np.block([block[i*block_incr:(i+1)*block_incr]])
        new_image = np.vstack([new_image, b])
    return new_image


"""
-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
Compression:
-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
"""
#Convert to textured image the input image
def ColorToTexturedImage(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    y, cr, cb = cv2.split(ycrcb)
    ycrcb = cv2.merge((y, cr, cb))
    plt.imshow(ycrcb)
    
    titles = ['L', ' H']
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
    
    titles = ['LL', ' LH', 'HL', 'HH']
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
    scipy.misc.imsave('textured.png', img)
    return img

#Permute the blocks randomly
def ShuffleImage(img, seed):
    block = []
    for i in range(0, factor, incr):
        for j in range(0, factor, incr):
            block.append(img[i:i+incr, j:j+incr])
    
    random.seed(seed)
    random.shuffle(block)
    
    new_image = ConstructImage(block)
    scipy.misc.imsave('textured_shuffled.png', new_image)
    return new_image, block

#Triangular shuffle the blocks
def TriangularShuffling(block):
    for i in range(0, len(block), 2):   #even
        block[i] = LtoRShift(block[i])
    for i in range(1, len(block), 2):   #odd
        block[i] = RtoLShift(block[i])
    return block

def Compression(img, seed):
    img = ColorToTexturedImage(img)
    new_image, block = ShuffleImage(img, seed)
    block = TriangularShuffling(block)
    new_image = ConstructImage(block)
    scipy.misc.imsave('textured_shuffled_triangular.png', new_image)
    
    
"""
-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
Decompression:
-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
"""
#Triangular reshuffle the image
def InvTriangularShuffling(img):
    block = []
    for i in range(0, factor, incr):
        for j in range(0, factor, incr):
            block.append(img[i:i+incr, j:j+incr])
    block = TriangularShuffling(block)
    new_image = ConstructImage(block)
    scipy.misc.imsave('tringular_textured_shuffled_triangular_rev1.png', new_image)
    return block

#Reorder the randomly permuted image
def ReshuffleImage(block, seed):
    new_block = [0]*len(block)
    Order = list(range(len(block)))
    random.seed(seed)
    random.shuffle(Order)
    for index, originalIndex in enumerate(Order):
        new_block[originalIndex] = block[index]
    new_image = ConstructImage(new_block)
    scipy.misc.imsave('tringular_textured_shuffled_triangular_rev2.png', new_image)
    return new_image

#Obtain the colored image from the textured image
def TexturedToColorImage(img):
    cfs = pywt.dwt(img, 'haar')
    CA, CB = cfs
    coeffs2 = pywt.dwt2(img, 'haar')
    LL, (LH, HL, HH) = coeffs2
    coeffs = (LL, (LH, HL, HH))
    y = pywt.idwt2(coeffs, 'haar')
    y = y.astype(np.uint8)
    new_img = cv2.merge((y, cr, cb))
    rgb = cv2.cvtColor(new_img, cv2.COLOR_YCR_CB2BGR)
    scipy.misc.imsave('tringular_textured_shuffled_triangular_rev3.png', rgb)

def Decompression(seed):
    img = cv2.imread('textured_shuffled_triangular.png', cv2.IMREAD_GRAYSCALE)
    block = InvTriangularShuffling(img)
    new_image = ReshuffleImage(block, seed)
    TexturedToColorImage(new_image)


img_name = "img07"
incr = 8
basic_seed = 63
org_img = cv2.imread(img_name + ".jpg", cv2.IMREAD_COLOR)
factor = min(org_img.shape[0], org_img.shape[1]) % incr
factor = min(org_img.shape[0], org_img.shape[1]) - factor
block_incr = factor//incr
img = org_img[:factor, :factor]
scipy.misc.imsave('original.png', img)
ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
y, cr, cb = cv2.split(ycrcb)
Compression(img, basic_seed)
Decompression(basic_seed)
