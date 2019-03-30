import numpy as np
import math
from matplotlib import pyplot as plt
import cv2 as cv
def histMatch(srcimg, tarimg, ftype='img', save=False):
	if ftype == 'file':
		src = cv.imread(srcimg, 1)
		tar = cv.imread(tarimg ,1)
	if ftype == 'img':
		src = srcimg
		tar = tarimg
	# the input must be RGB img
	def oneChannelMatch(srcChannel ,tarChannel, level=256):
		# the input channel shape is (256,)
		shape = srcChannel.shape
		srcChannel_c = srcChannel.flatten()
		tarChannel_c = tarChannel.flatten()
		MN = srcChannel_c.shape[0]
		srcHist, _ = np.histogram(srcChannel_c, level, (0, 255))
		tarHist, _ = np.histogram(tarChannel_c, level, (0, 255))
		srcHist = srcHist.astype(np.float)
		tarHist = tarHist.astype(np.float)
		stemp = 0
		ttemp = 0
		for i in range(level):
			stemp += srcHist[i] / MN
			ttemp += tarHist[i] / MN
			srcHist[i] = stemp
			tarHist[i] = ttemp
			# print(ttemp)
		# exit()
		table = np.zeros(level)
		for sindex, srchist in enumerate(srcHist):
			index = 0
			minVal = 1
			for tindex, tarhist in enumerate(tarHist):
				if np.fabs(tarhist - srchist) < minVal:
					minVal = np.fabs(tarhist - srchist)
					index = tindex
			table[sindex] = index
		# exit()
		return table[srcChannel]
	
	sB, sG, sR = src[:,:,0], src[:,:,1], src[:,:,2]
	tB, tG, tR = tar[:,:,0], tar[:,:,1], tar[:,:,2]
	rB = oneChannelMatch(sB, tB)
	rG = oneChannelMatch(sG, tG)
	rR = oneChannelMatch(sR, tR)
	img = cv.merge([rB, rG, rR])
	if save and ftype == 'file':
		cv.imwrite(srcimg, img)
	if not save:
		return img

if __name__ == '__main__':
	import cv2 as cv
	src = cv.imread('./luoxing/1/1_127_019.jpg', 1)
	tar = cv.imread('./luoxing/0/1_127_019.jpg', 1)
	img = histMatch(src, tar)
	print(img.shape)
	cv.imwrite('result.jpg', img)