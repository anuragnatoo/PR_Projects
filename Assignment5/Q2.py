from skimage.exposure import rescale_intensity
import numpy as np
import cv2

def convolve(image, kernel):
	(iH, iW) = image.shape[:2]
	(kH, kW) = kernel.shape[:2]

	
	pad = (kW - 1) // 2
	image = cv2.copyMakeBorder(image, pad, pad, pad, pad,
		cv2.BORDER_REPLICATE)
	output = np.zeros((iH, iW), dtype="float32")

	
	for y in np.arange(pad, iH + pad):
		for x in np.arange(pad, iW + pad):
		
			roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]

		
			k = (roi * kernel).sum()
			output[y - pad, x - pad] = k

	#output = rescale_intensity(output, in_range=(0, 255))
	output = (output * 255).astype("uint8")

	# return the output image
	return output
kernel1 = np.array((
	[1, 1, 1],
	[0, 0, 0],
	[-1, -1, -1]), dtype="int")

kernel2 = np.array((
	[-1, 0, 1],
	[-1, 0, 1],
	[-1, 0, 1]), dtype="int")

def convoLve(image,kernel):
 z=cv2.filter2D(image, -1, kernel)
 return z


kernelBank = (
	("kernel1",kernel1),
	("kernel2",kernel2),
)

# load the input image
image = cv2.imread('baboon.png')

# loop over the kernels
for (kernelName, kernel) in kernelBank:
	print("applying {} ".format(kernelName))
	convoleOutput = convoLve(image, kernel)

	# show the output images
	cv2.imshow("original", image)
	cv2.imshow("{} - convole".format(kernelName), convoleOutput)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

