import json

class Config():
	def __init__(self, confPath):
		confAll = json.loads(open(confPath).read())
		self.__output = confAll['output']
		self.__oldImgPath = confAll['old_img_path']
		self.__newImgPath = confAll['new_img_path']
		self.__maskPath = confAll['mask_path']
		self.__size = confAll['size']

	def getOutputPath(self):
		return self.__output

	def getOldPath(self):
		return self.__oldImgPath

	def getNewPath(self):
		return self.__newImgPath

	def getMaskPath(self):
		return self.__maskPath

	def getSize(self):
		return self.__size