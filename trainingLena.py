import Image_restoration_Wiener_Filter as I
import numpy as np
import cv2


K=I.findK()
sigma=I.findsigma()
size=I.findsize()

#print size




f=I.openImage('lena.png')
I.printImage("original image",f)


# In[83]:


f=I.grayImage(f)
I.printImage("gray image",f)


# In[84]:


psf=I.PSF(size,sigma)
I.printImage("Point Spread Function",psf)


# In[85]:


noise = I.NOISE(0,200,f.shape)
I.printImage("Noise",noise)


# In[86]:


blurred = I.MakeBlurred(f,psf)
I.printImage("Blurred Image",blurred)


# In[87]:


g=I.blurNnoisy(blurred,noise)
I.printImage("Blurred and Noisy Image",g)


# In[88]:


fhat = I.weinerFilter(f,g,psf,K)
I.printImage("Restored Image",fhat)


# In[89]:


truesnr=np.mean(I.TrueSNR(f,noise))
print "SNR : ",str(truesnr)
print "actual K : ",str(1/truesnr)

cv2.waitKey(0)
cv2.destroyAllWindows()

K=np.save("saveK.npy",K)
psf=np.save("savepsf.npy",psf)
#sigma=np.save("savesigma.npy",sigma)
#size=np.save("savesize.npy",size)