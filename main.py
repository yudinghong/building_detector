from train import TrainModel
from process import Preprocess

pre = Preprocess('config.json')
pre.preprocess()
model = TrainModel('config.json')
model.train()