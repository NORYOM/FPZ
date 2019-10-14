import numpy as np
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.externals import joblib

class Classify:

	X = []	# 训练数据集
	y = []	# 类型数据集
	clsf = None

	def  __init__(self):
		self.clsf = svm.SVC(gamma='scale')

	def setTrainData(self,trainData):
		for t in trainData:
			self.X.append(t)

	def setTypeData(self,typeData):
		for t in typeData:
			self.y.append(t)

	def train(self):
		#### 待完成!!! self.X 需要降维处理，否则不能进行训练
		#  1、尝试使用 svc = LinearSVC() 或者
		#  2、先将 32*32 图形矩阵降维至 1*1024再分类
		self.clsf.fit(self.X,self.y)

	def chkType(self,targetData):
		arr = self.predict([targetData])
		return arr[0]

	def saveModule(self):
		joblib.dump(self.clsf,'./train.mdl')
