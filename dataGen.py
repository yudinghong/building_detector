import numpy as np
import cv2 as cv
import os
import json
import math

import matplotlib.pyplot as plt
# create data batch, concatenate the old img and the new img
# 
class DataGen():
	def __init__(self, oldImgPath, newImgPath, fileType='jpg'):
		# fileType can be tif or jpg or png
		from glob import glob
		self.__oldPath = oldImgPath
		self.__newPath = newImgPath
		self.__oldImgs = np.array(glob(oldImgPath+'/*.'+fileType))
		self.__namelist = []
		for one in self.__oldImgs:
			self.__namelist.append(one.split('\\')[1].split('.')[0])
		self.__namelist = np.array(self.__namelist)
		np.random.shuffle(self.__namelist)
		self.__dataNum = self.__namelist.shape[0]
		self.__trainNum = math.floor(self.__dataNum * 0.8)
		self.__validateNum = self.__dataNum - self.__trainNum
		self.__trainList = self.__namelist[0:self.__trainNum]
		self.__validateList = self.__namelist[self.__trainNum:]

	def getData(self, flag='train'):
		start = 0
		print(flag)
		if flag == 'train':
			namelist = self.__trainList
			dataNum = self.__trainNum
		elif flag == 'validate':
			namelist = self.__validateList
			dataNum = self.__validateNum
		else:
			namelist = self.__trainList
			dataNum = self.__trainNum
		i = 0
		index = 0
		while True:
			# get the one data from file
			oneResult = self.__readImg(namelist[index])
			if start == 0:
				temp = oneResult[np.newaxis,:,:,:]
				start = 1
			else:
				temp = np.concatenate([temp, oneResult[np.newaxis,:,:,:]])
			i += 1
			index = (index + 1) % dataNum
			if i % 32 == 0:
				start = 0
				yield (temp[:,:,:, 0:6], temp[:,:,:,6:7])
				

	def __readImg(self, name):
		oldImg = cv.imread(self.__oldPath+'/'+name+'.jpg', 1)
		newImg = cv.imread(self.__newPath+'/'+name+'.jpg', 1)
		if oldImg.shape != newImg.shape:
			print(name + ' shape not match')
			return -1
		x, y, z = oldImg.shape
		zeroMask = np.zeros([x, y])
		if os.path.exists(self.__newPath+'/'+name+'.json'):
			try:
				conf = json.loads(open(self.__newPath+'/'+name+'.json').read())
				shapes = conf['shapes']
				for shape in shapes:
					point = np.array(shape['points'])
					point = point[np.newaxis, :, :]
					cv.polylines(zeroMask, point, 1, 1)
					cv.fillPoly(zeroMask, point, 1)
			except:
				print('no json text')
			
		zeroMask = zeroMask[:,:, np.newaxis]
		return np.concatenate([oldImg, newImg, zeroMask], axis=2)
			# print(conf)

	def getTrainNum(self):
		return self.__trainNum

	def getValidateNum(self):
		return self.__validateNum

if __name__ == '__main__':
	data = DataGen('luoxing/0', 'luoxing/1')