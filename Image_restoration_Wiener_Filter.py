#!/usr/bin/env python
# coding: utf-8

# In[69]:


import cv2
import os 
#from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import convolve2d 
from scipy.ndimage import gaussian_filter
from sklearn.metrics import mean_squared_error
import math
#import trainingLena as Lena


# In[70]:


def openImage(image_name):
    f = cv2.imread(image_name)
    return f


# In[71]:


def grayImage(f):
    f = cv2.cvtColor(f,cv2.COLOR_BGR2GRAY)
    #plt.gray()
    return f


# In[72]:


def printImage(string,func):
    print func.shape
    #plt.xlabel(string)
    #plt.imshow(func)
    cv2.imshow(string,func)


# In[73]:


def PSF(size,sigma):
    PSF=np.zeros((size,size))
    PSF[size//2,size//2]=1
    h = gaussian_filter(PSF, sigma)
    return h


# In[74]:


def NOISE(mean,variance,Shape):
    standarddev=variance**0.5
    n=np.random.normal(mean,standarddev,(Shape))
    return n


# In[75]:


def MakeBlurred(f,h):
    blurred= convolve2d(f, h, 'same')
    blurred=np.uint8(blurred)
    return blurred


# In[76]:


def blurNnoisy(blurred,n):
    g = blurred + n
    g=np.uint8(g)
    return g


# In[77]:


def calculateRestored(g,h,K):
    G=np.fft.fft2(g)
    #N=np.fft.fft2(n)
    H=np.fft.fft2(h,g.shape)
    conjH=np.matrix.conjugate(H)
    
    Hsquare=np.multiply(H,conjH)
    W=np.divide(conjH,(Hsquare+K))
    Fhat=np.multiply(W,G)
    fhat=np.fft.ifft2(Fhat)
    fhat=np.abs(fhat)
    fhat=np.uint8(fhat)
    return fhat


# In[78]:


def MSE(fhat,f):
    mse = mean_squared_error(fhat, f)
    #print mse
    return mse


# In[79]:


def PSNR(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 10
    PIXEL_MAX = 255.0
    psnr = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    return psnr


# In[80]:


def weinerFilter(f,g,h,K):
    fhat=calculateRestored(g,h,K)

    mse = MSE(fhat,f)
    print "MSE : %f"%(mse)
    psnr=PSNR(fhat,f)
    print "PSNR : %f"%(psnr)
    return fhat


# In[81]:


def TrueSNR(f,n):
    F=np.fft.fft2(f)
    N=np.fft.fft2(n)
    conjF=np.matrix.conjugate(F)
    conjN=np.matrix.conjugate(N)
    SNN=np.multiply(N,conjN)
    SFF=np.multiply(F,conjF)
    SNR=np.divide(SNN,SFF)
    SNR=np.abs(SNR)
    return SNR

def findK():
	print "optimum value of K = ",str(0.508)," (default)"
	K=raw_input("Enter a value for K : ")
	if K:
	    K=float(K)
	else :
	    K=0.508 ## optimum K for lena
	return K

def findsigma():
	print "optimum value of sigma for PSF = ",str(1)," (default)"
	sigma=raw_input("Enter a value for sigma for PSF : ")
	if sigma:
	    sigma=int(sigma)
	else :
	    sigma=1 ## optimum sigma for lena
	return sigma

def findsize():
	print "optimum value of size for PSF = ",str(5)," (default)"
	size=raw_input("Enter a value for size for PSF : ")
	if size:
	    size=int(size)
	else :
	    size=5 ## optimum size for lena
	return size


# ans=raw_input("Enter 1 for Train, 2 for Test ")

# if(ans=="1"):
# 	K=findK()
# 	sigma=findsigma()

# 	f=openImage('lena.png')
# 	printImage("original image",f)

# 	f=grayImage(f)
# 	printImage("gray image",f)

# 	psf=PSF(5,sigma)
# 	printImage("Point Spread Function",psf)

# 	noise = NOISE(0,200,f.shape)
# 	printImage("Noise",noise)

# 	blurred = MakeBlurred(f,psf)
# 	printImage("Blurred Image",blurred)

# 	g=blurNnoisy(blurred,noise)
# 	printImage("Blurred and Noisy Image",g)

# 	fhat = weinerFilter(f,g,psf,K)
# 	printImage("Restored Image",fhat)

# 	truesnr=np.mean(TrueSNR(f,noise))
# 	print "SNR : ",str(truesnr)
# 	print "actual K : ",str(1/truesnr)

# 	cv2.waitKey(0)
# 	cv2.destroyAllWindows()

# else: #if (ans=="n")

# 	sigma=1
# 	print "value of sigma = ",str(sigma)," (default)"
# 	psf=PSF(5,sigma)
# 	K=0.508
# 	print "value of K = ",str(K)," (default)"
	



# # Training

# In[82]:


# f=openImage('lena.png')
# printImage("original image",f)


# # In[83]:


# f=grayImage(f)
# printImage("gray image",f)


# # In[84]:


# psf=PSF(5,1)
# printImage("Point Spread Function",psf)


# # In[85]:


# noise = NOISE(0,200,f.shape)
# printImage("Noise",noise)


# # In[86]:


# blurred = MakeBlurred(f,psf)
# printImage("Blurred Image",blurred)


# # In[87]:


# g=blurNnoisy(blurred,noise)
# printImage("Blurred and Noisy Image",g)


# # In[88]:


# fhat = weinerFilter(f,g,psf,0.5008)
# printImage("Restored Image",fhat)


# # In[89]:


# truesnr=np.mean(TrueSNR(f,noise))
# print "SNR : ",str(truesnr)
# print "actual K : ",str(1/truesnr)


# # Test

# ##  1. Harry
# 

# In[90]:


# f_harry=openImage('harry.png')
# printImage("original image",f_harry)


# # In[91]:


# f_harry=grayImage(f_harry)
# printImage("gray image",f_harry)


# # In[92]:


# n_harry = NOISE(0,100,f_harry.shape)
# printImage("Noise",n_harry)


# # In[93]:


# h_harry=PSF(10,10)
# printImage("Point Spread Function",h_harry)


# # In[94]:


# blurred_harry = MakeBlurred(f_harry,h_harry)
# printImage("Blurred Image",blurred_harry)


# # ### Test image : harry

# # In[95]:


# n_harry=cv2.resize(n_harry,f_harry.shape)
# g_harry=blurNnoisy(blurred_harry,n_harry)
# printImage("Blurred and Noisy Image",g_harry)


# # ### Applying wiener filter on test image

# # In[96]:


# fhat_harry =weinerFilter(f_harry,g_harry,psf,noise,0.112)
# printImage("Restored Image",fhat_harry)


# # In[97]:


# truesnr=np.mean(TrueSNR(f,noise))
# print "SNR : ",str(truesnr)
# print "actual K : ",str(1/truesnr)


# # ## 2. Car Number Plate

# # In[98]:


# f_car=openImage('car.jpg')
# printImage("original image",f_car)


# # In[99]:


# f_car=grayImage(f_car)
# printImage("gray image",f_car)


# # In[100]:


# n_car = NOISE(0,300,f_car.shape)
# printImage("Noise",n_car)


# # In[101]:


# h_car=PSF(5,5)
# printImage("Point Spread Function",h_car)


# # In[102]:


# blurred_car = MakeBlurred(f_car,h_car)
# printImage("Blurred Image",blurred_car)


# # ### Test Image : car

# # In[103]:


# n_car=cv2.resize(n_car,f_car.shape)
# g_car=blurNnoisy(blurred_car,n_car)
# printImage("Blurred and Noisy Imag_care",g_car)


# # ### Applying wiener filter on test image

# # In[104]:


# fhat_car =weinerFilter(f_car,g_car,psf,noise,0.9)
# printImage("Restored Image",fhat_car)


# # In[105]:


# truesnr=np.mean(TrueSNR(f,noise))
# print "SNR : ",str(truesnr)
# print "actual K : ",str(1/truesnr)


# # ## 3. Street Sign

# # In[106]:


# f_sign=openImage('streetsign1.jpg')
# printImage("original image",f_sign)


# # In[107]:


# f_sign=grayImage(f_sign)
# printImage("gray image",f_sign)


# # In[108]:


# f_sign=cv2.resize(f_sign,(f_sign.shape[1],f_sign.shape[1]))
# printImage("gray image",f_sign)


# # In[109]:


# n_sign = NOISE(0,1000,f_sign.shape)
# printImage("Noise",n_sign)


# # In[110]:


# h_sign=PSF(5,5)
# printImage("Point Spread Function",h_sign)


# # In[111]:


# blurred_sign = MakeBlurred(f_sign,h_sign)
# printImage("Blurred Image",blurred_sign)


# # ### Test image : street sign

# # In[112]:


# n_sign=cv2.resize(n_sign,f_sign.shape)

# g_sign=blurNnoisy(blurred_sign,n_sign)
# printImage("Blurred and Noisy Image",g_sign)


# # ### Applying wiener filter on test image

# # In[113]:


# fhat_sign =weinerFilter(f_sign,g_sign,psf,noise,0.518)
# printImage("Restored Image",fhat_sign)


# # In[114]:


# truesnr=np.mean(TrueSNR(f,noise))
# print "SNR : ",str(truesnr)
# print "actual K : ",str(1/truesnr)


# In[ ]:




# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




