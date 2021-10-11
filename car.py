import Image_restoration_Wiener_Filter as I
import numpy as np
import cv2

# Test

##  1. car


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

f_car=I.openImage('car.jpg')
I.printImage("original image",f_car)


# In[91]:


f_car=I.grayImage(f_car)
I.printImage("gray image",f_car)


# In[92]:


n_car = I.NOISE(0,100,f_car.shape)
I.printImage("Noise",n_car)


# In[93]:


h_car=I.PSF(5,5)
I.printImage("Point Spread Function",h_car)


# In[94]:


blurred_car = I.MakeBlurred(f_car,h_car)
I.printImage("Blurred Image",blurred_car)


# ### Test image : car

# In[95]:


n_car=cv2.resize(n_car,f_car.shape)
g_car=I.blurNnoisy(blurred_car,n_car)
I.printImage("Blurred and Noisy Image",g_car)


# ### Applying wiener filter on test image

# In[96]:
print "K used for testing : ",str(K)
fhat_car =I.weinerFilter(f_car,g_car,psf,K) 
I.printImage("Restored Image",fhat_car)


# In[97]:


truesnr=np.mean(I.TrueSNR(f_car,n_car))
print "SNR : ",str(truesnr)
print "actual K : ",str(1/truesnr)

cv2.waitKey(0)
cv2.destroyAllWindows()