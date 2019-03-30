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
		srcChannel = srcChannel.flatten()
		tarChannel = tarChannel.flatten()
		MN = srcChannel.shape[0]
		srcHist, _ = np.histogram(srcChannel, level)
		tarHist, _ = np.histogram(tarChannel, level)

		def countTable(hist):
			count = 0
			table = [0]*256
			for index, data in enumerate(hist):
				count += data
				p = count / MN * level
				table[index] = round(p)
			return table

		srcTab = countTable(srcHist)
		tarTab = countTable(tarHist)
		resTab = [0]*256
		tempIndex = 0
		for sindex, sdata in enumerate(srcTab):
			for tindex, tdata in enumerate(tarTab):
				if sdata <= tdata:
					if sdata < (tdata + tarTab[tindex-1])/2 and tindex > 0:
						tempIndex = tindex-1
					if sdata >= (tdata + tarTab[tindex-1])/2 and tindex > 0: 
						tempIndex = tindex
					if tindex == 0:
						tempIndex = 0
					break
			resTab[sindex] = tempIndex
			print(resTab[sindex])
		result = []
		for one in srcChannel:
			result.append(resTab[one])
		result = np.array(result)
		result = np.reshape(result, shape)
		result = result.astype(np.uint8)
		return result
	
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
	src = cv.imread('./new3.tif', 1)
	tar = cv.imread('./old3.tif', 1)
	histMatch(src, tar)