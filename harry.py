import Image_restoration_Wiener_Filter as I
import numpy as np
import cv2


# Test

##  1. Harry


#In[90]:

# K=raw_input("Enter a value for K : ")
# if K:
#     K=float(K)
# else :
#     K=0.508


K=np.load("saveK.npy")
#sigma=np.load("savesigma.npy")
#size=np.load("savesize.npy")
psf=np.load("savepsf.npy")

f_harry=I.openImage('harry.png')
I.printImage("original image",f_harry)


# In[91]:


f_harry=I.grayImage(f_harry)
I.printImage("gray image",f_harry)


# In[92]:


n_harry = I.NOISE(0,50,f_harry.shape)
I.printImage("Noise",n_harry)


# In[93]:


h_harry=I.PSF(5,5)
I.printImage("Point Spread Function",h_harry)


# In[94]:


blurred_harry = I.MakeBlurred(f_harry,h_harry)
I.printImage("Blurred Image",blurred_harry)


# ### Test image : harry

# In[95]:


n_harry=cv2.resize(n_harry,f_harry.shape)
g_harry=I.blurNnoisy(blurred_harry,n_harry)
I.printImage("Blurred and Noisy Image",g_harry)


# ### Applying wiener filter on test image

# In[96]:

print "K used for testing : ",str(K)
fhat_harry =I.weinerFilter(f_harry,g_harry,psf,K)
I.printImage("Restored Image",fhat_harry)


# In[97]:


truesnr=np.mean(I.TrueSNR(f_harry,n_harry))
print "SNR : ",str(truesnr)
print "actual K : ",str(1/truesnr)

cv2.waitKey(0)
cv2.destroyAllWindows()