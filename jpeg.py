from matplotlib.image import imread
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['figure.figsize'] = [5,5]
plt.rcParams.update({'font.size':10})

A = imread(os.path.join('/home/vinay/SEM1/MATHS/MathsA2','girl.jpg'))
B = np.mean(A,-1); #Convert RGB to Gray Scale by averaging over 3 channels

dim = A.shape
print('Original Dimensions:',dim)

plt.figure()
plt.imshow(A)
plt.axis('off')
#plt.show()


Bt = np.fft.fft2(B) # 2D FFT on image B to get fourier coefficient matrix Bt
Btsort = np.sort(np.abs(Bt.reshape(-1)))  #sort by magnitude after stretching Mat Bt in vector

for keep in (0.1, 0.05, 0.01, 0.002):
	thresh = Btsort[int(np.floor((1-keep)*len(Btsort)))] #creating threshold
	ind = np.abs(Bt)>thresh #removing coeficients less than threshold - ind matrix
	Btlow = Bt *ind  #multiplying by 0  and 1
	Alow = np.fft.ifft2(Btlow).real #Alow is low pass filtered matrix -> IFFT (compressed image)
	plt.figure()
	plt.imshow(Alow,cmap = 'gray')
	plt.axis('off')
	plt.title('Compressed image: keeping = '+ str(keep*100) + '%')
	
plt.show()

	
	
	
	
	
	
	
