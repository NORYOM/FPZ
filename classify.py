from sklearn import svm

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
		self.clsf.fit(X,y)

	def chkType(self,targetData):
		arr = self.predict([targetData])
		return arr[0]