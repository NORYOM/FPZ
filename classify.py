import numpy as np
from sklearn import svm
from sklearn.externals import joblib
from sklearn import tree
# from sklearn import neighbors

class Classify:

	X = []	# 训练数据集
	y = []	# 类型数据集
	clsf = None
	tsne = None

	def  __init__(self):
		# self.clsf = svm.SVC(gamma='scale')
		self.clsf = tree.DecisionTreeClassifier()
		# self.clsf = neighbors.KNeighborsClassifier(30, weights='distance')

	def setTrainData(self,trainData):
		self.X.extend(trainData)

	def setTypeData(self,typeData):
		self.y.extend(typeData)

	def train(self):
		if len(self.y)<2:
			print("样本少于 2 个，请继续选择")
			return
		print("开始训练")

		self.clsf.fit(self.X,self.y)
		print("训练完成，共 %d 个正例 %d 个反例。" % (self.y.count(1), self.y.count(0)))

	def chkType(self,faceData):
		clsf = joblib.load('./train.mdl')
		arr = clsf.predict(faceData)
		score = clsf.predict_proba(faceData) # 只有当使用决策树时才用，其他方式需要修改返回的 score 为 None
		# score = None
		return arr,score

	def saveModule(self):
		print("开始保存模型")
		joblib.dump(self.clsf,'./train.mdl')
		print("保存结束")
