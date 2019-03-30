from keras.optimizers import Adam
from keras.callbacks import Callback, ModelCheckpoint, CSVLogger, LearningRateScheduler
from dataGen import DataGen
import keras.backend as K
import math
import os

def f1_score(y_true, y_pred, smooth=1):
	"""
	f1 score，用于训练过程中选择模型
	"""
	y_true = y_true[:,:,:,-1] #通道？
	y_pred = y_pred[:,:,:,-1] #
	c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	c2 = K.sum(K.round(K.clip(y_pred, 0, 1))) #round 四舍五入 clip将超过范围的值强制转换为边界值
	c3 = K.sum(K.round(K.clip(y_true, 0, 1)))
	f1_score = (2*c1+smooth)/(c2+c3+smooth)
	return f1_score

def dice_coef(y_true, y_pred, smooth=1, weight=1):
	"""
	加权后的dice coefficient
	"""
	y_true = y_true[:,:,:,-1]
	y_pred = y_pred[:,:,:,-1]
	intersection = K.sum(y_true * y_pred)
	union = K.sum(y_true) + weight*K.sum(y_pred)
	return (2. * intersection + smooth) / (union + smooth)

def dice_coef_loss(y_true, y_pred):
	"""
	目标函数
	"""
	return 1-dice_coef(y_true, y_pred)

class TrainModel():
	def __init__(self, confPath):
		self.__output = ''
		self.__newInput = ''
		self.__oldInput = ''
		self.__size = 0
		self.__train = []
		self.__validate = []
		self.__newimg = []
		self.__oldimg = []
		self.__maskPath = []
		self.__setConf(confPath)
		import unet
		self.__model = unet.getUnetBySize(self.__size, (128, 128, 6), 1)
		return
	
	def train(self):
		metrics = [f1_score]
		optimizer = Adam(lr=1e-5, decay=0)
		loss_function = dice_coef_loss
		callbacks = self.__makeCallbacks()
		self.__model.compile(optimizer, loss=loss_function, metrics=metrics)
		data = DataGen('luoxing/0', 'luoxing/1')
		trainData = data.getData(flag='train')
		validateData = data.getData(flag='validate')
		self.__model.fit_generator(trainData, verbose=1, validation_data=validateData, steps_per_epoch=math.ceil(data.getTrainNum()/32), 
		validation_steps=math.ceil(data.getValidateNum()/32), epochs=100, callbacks=callbacks)
		return 

	# set the callback func
	def __makeCallbacks(self):
		best_model_checkpoint = ModelCheckpoint(monitor='val_f1_score', mode='max', filepath='model.h5', save_best_only=True, save_weights_only=True)
		logger = CSVLogger('log.txt', append=True)
		callbacks = [best_model_checkpoint, logger]
		# def get_lr(epoch):
		# 	w = epoch // 10
		# 	lr = 1e-5 / (10 ** w)
		# 	if lr < 1e-10:
		# 		lr = 1e-10
		# 		return lr
		# callback = LearningRateScheduler(get_lr)
		# callbacks.append(callback)

		return callbacks

	# set the config about the train
	def __setConf(self, confPath):
		from config import Config
		conf = Config('config.json')
		self.__newimg = conf.getNewPath()
		self.__oldimg = conf.getOldPath()
		self.__size = conf.getSize()
		self.__maskPath = conf.getMaskPath()
		self.__output = conf.getOutputPath()
		del conf
		del Config

if __name__ == '__main__':
	model = TrainModel('config.json')
	model.train( )