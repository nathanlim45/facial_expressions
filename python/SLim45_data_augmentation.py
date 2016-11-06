
# coding: utf-8

# In[1]:

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

os.chdir('/Users/nathanlim/face_image')


# ### Image Tranformation

# In[3]:

import cv2
import glob
num = 1
for img in glob.glob("/Users/nathanlim/face_image/*.jpg"):
    cv_img = cv2.imread(img)
    rows,cols,ch = cv_img.shape

#Affine
    pts1 = np.float32([[50,50],[200,50],[50,200]])
    pts2 = np.float32([[52,50],[202,50],[48,200]])
    pts3 = np.float32([[54,50],[204,50],[46,200]])
    pts4 = np.float32([[56,50],[206,50],[44,200]])
    pts5 = np.float32([[58,50],[208,50],[42,200]])
    pts6 = np.float32([[60,50],[210,50],[40,200]])
    
    M1 = cv2.getAffineTransform(pts1,pts2)
    M2 = cv2.getAffineTransform(pts1,pts3)
    M3 = cv2.getAffineTransform(pts1,pts4)
    M4 = cv2.getAffineTransform(pts1,pts5)
    M5 = cv2.getAffineTransform(pts1,pts6)
#angle
    M6 = cv2.getRotationMatrix2D((cols/2,rows/2),10,1)
    M7 = cv2.getRotationMatrix2D((cols/2,rows/2),-10,1)
    M8 = cv2.getRotationMatrix2D((cols/2,rows/2),20,1)
    M9 = cv2.getRotationMatrix2D((cols/2,rows/2),-20,1)

    dst1 = cv2.warpAffine(cv_img,M1,(cols,rows))
    dst2 = cv2.warpAffine(cv_img,M2,(cols,rows))
    dst3 = cv2.warpAffine(cv_img,M3,(cols,rows))
    dst4 = cv2.warpAffine(cv_img,M4,(cols,rows))
    dst5 = cv2.warpAffine(cv_img,M5,(cols,rows))
    dst6 = cv2.warpAffine(cv_img,M6,(cols,rows))
    dst7 = cv2.warpAffine(cv_img,M7,(cols,rows))
    dst8 = cv2.warpAffine(cv_img,M8,(cols,rows))
    dst9 = cv2.warpAffine(cv_img,M9,(cols,rows))
    cv2.imwrite('/Users/nathanlim/face_image/output/type1_'+str(num)+'.jpg', dst1)
    cv2.imwrite('/Users/nathanlim/face_image/output/type2_'+str(num)+'.jpg', dst2)
    cv2.imwrite('/Users/nathanlim/face_image/output/type3_'+str(num)+'.jpg', dst3)
    cv2.imwrite('/Users/nathanlim/face_image/output/type4_'+str(num)+'.jpg', dst4)
    cv2.imwrite('/Users/nathanlim/face_image/output/type5_'+str(num)+'.jpg', dst5)
    cv2.imwrite('/Users/nathanlim/face_image/output/type6_'+str(num)+'.jpg', dst6)
    cv2.imwrite('/Users/nathanlim/face_image/output/type7_'+str(num)+'.jpg', dst7)
    cv2.imwrite('/Users/nathanlim/face_image/output/type8_'+str(num)+'.jpg', dst8)
    cv2.imwrite('/Users/nathanlim/face_image/output/type9_'+str(num)+'.jpg', dst9)
    num += 1
#   plt.subplot(1,10,1),plt.imshow(cv_img),plt.title('Input')
#   plt.subplot(1,10,2),plt.imshow(dst1)
#   plt.subplot(1,10,3),plt.imshow(dst2)
#   plt.subplot(1,10,4),plt.imshow(dst3)
#   plt.subplot(1,10,5),plt.imshow(dst4)
#   plt.subplot(1,10,6),plt.imshow(dst5)
#   plt.subplot(1,10,7),plt.imshow(dst6)
#   plt.subplot(1,10,8),plt.imshow(dst7)
#   plt.subplot(1,10,9),plt.imshow(dst8)
#   plt.subplot(1,10,10),plt.imshow(dst9)
#   
#   plt.show()


# ### Example

# In[6]:

img = cv2.imread('face_image0004.jpg')
rows,cols,ch = img.shape

pts1 = np.float32([[50,50],[200,50],[50,200]])
pts2 = np.float32([[60,50],[210,50],[40,200]])

M = cv2.getAffineTransform(pts1,pts2)
M2 = cv2.getRotationMatrix2D((cols/2,rows/2),10,1)
img_scaled = cv2.resize(img,None,fx=1.5, fy=1, interpolation = cv2.INTER_LINEAR) #resize

dst = cv2.warpAffine(img,M,(cols,rows))
dst2 = cv2.warpAffine(img,M2,(cols,rows))

plt.subplot(141),plt.imshow(img),plt.title('Input')
plt.subplot(142),plt.imshow(dst),plt.title('exe1')
plt.subplot(143),plt.imshow(dst2),plt.title('exe2')
plt.subplot(144),plt.imshow(img_scaled),plt.title('Resizing')

plt.show()


# ## Transformation method
# 
# Mostly, I use Affine transformation and way to change the angle. I used 9 different parameters and get 5000 pictures including the original.
# I excluded the resizing method for the convience reason to apply CNN for the future.

# In[ ]:



