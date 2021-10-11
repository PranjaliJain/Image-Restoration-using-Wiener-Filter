import Image_restoration_Wiener_Filter as I
import numpy as np
import cv2

# Test

##  1. sign


#In[90]:

# K=raw_input("Enter a value for K : ")
# if K:
#     K=float(K)
# else :
#     K=0.15

K=np.load("saveK.npy")
#sigma=np.load("savesigma.npy")
#size=np.load("savesize.npy")
psf=np.load("savepsf.npy")

f_sign=I.openImage('streetsign1.jpg')
I.printImage("original image",f_sign)


# In[91]:


f_sign=I.grayImage(f_sign)
#I.printImage("gray image",f_sign)

f_sign=cv2.resize(f_sign,(f_sign.shape[1],f_sign.shape[1]))
I.printImage("gray image",f_sign)
# In[92]:


n_sign = I.NOISE(0,100,f_sign.shape)
I.printImage("Noise",n_sign)


# In[93]:


h_sign=I.PSF(5,5)
I.printImage("Point Spread Function",h_sign)


# In[94]:


blurred_sign = I.MakeBlurred(f_sign,h_sign)
I.printImage("Blurred Image",blurred_sign)


# ### Test image : sign

# In[95]:


n_sign=cv2.resize(n_sign,f_sign.shape)
g_sign=I.blurNnoisy(blurred_sign,n_sign)
I.printImage("Blurred and Noisy Image",g_sign)


# ### Applying wiener filter on test image

# In[96]:

print "K used for testing : ",str(K)
fhat_sign =I.weinerFilter(f_sign,g_sign,psf,K) ## Change the parameter Kk from here 
I.printImage("Restored Image",fhat_sign)


# In[97]:


truesnr=np.mean(I.TrueSNR(f_sign,n_sign))
print "SNR : ",str(truesnr)
print "actual K : ",str(1/truesnr)

cv2.waitKey(0)
cv2.destroyAllWindows()