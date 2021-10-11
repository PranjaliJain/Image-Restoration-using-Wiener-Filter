# Image-Restoration-using-Wiener-Filter

This python implementation can be used to restore an image which is blurred and noisy to its original
form using the Wiener Filter algorithm.


## How to:
To run the code for Image Restoration using Wiener Filter, use jupyter notebook and open the
file Image_restoration_Wiener_Filter.ipynb and run the file.

The training of Wiener filter on Lena image can also be run using terminal
- > python Image_restoration_Wiener_Filter.py
- > python trainingLena.py

trainingLena.py prompts the user to train the filter using Lena image. The user can enter a
desired value for K and size and sigma for PSF to evaluate which K and PSF performs best
for the Lena image.

While testing the filter on other images the values of Point Spread Function and K remain
same as that for training Lena image.

The testing of the wiener filter can be done using terminal as follows :
- > python harry.py
- > python car.py
- > python streetsign.py

## Libraries Used:
- cv2
- os
- numpy
- scipy
- matplotlib

## Results:
See the report file for the results of tested images. 

#### Note: 
The images formed by running .py files through terminal have an additional conversion to
uint8(not present in .ipynb file) which mediates printing these images. Some of the test images
have noise inherently and on introducing a blur factor , white dots can be seen. However, these
images appear perfectly fine in the .ipynb file
