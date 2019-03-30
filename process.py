from hist import histMatch
from glob import glob
import numpy as np

class Preprocess():
	def __init__(self, config, ftype='jpg'):
		self.__newpath = ''
		self.__oldpath = ''
		self.__ftype = ftype
		self.__setConf(config)
		self.__namelist = []
		self.__newImgs = np.array(glob(self.__newpath+'/*.'+ftype))
		self.__oldImgs = np.array(glob(self.__oldpath+'/*.'+ftype))
		for one in self.__oldImgs:
			self.__namelist.append(one.split('\\')[1].split('.')[0])

	def preprocess(self):
		for one in self.__namelist:
			histMatch(self.__newpath+'/'+one+self.__ftype, self.__oldpath+'/'+one+self.__ftype, ftype='file', save=True)

	def __setConf(self, confPath):
		from config import Config
		conf = Config('config.json')
		self.__newpath = conf.getNewPath()
		self.__oldpath = conf.getOldPath()
		del conf
		del Config
