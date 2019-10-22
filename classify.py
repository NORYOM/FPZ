import numpy as np
# from sklearn import svm
from sklearn import manifold
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
		self.tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)

	def setTrainData(self,trainData):
		for t in trainData:
			self.X.append(t)

	def setTypeData(self,typeData):
		for t in typeData:
			self.y.append(t)

	# 降维处理
	def dimentionTransform(self,faceData,isTrain=False):
		newX = []
		faceCnt = 0
		for x in faceData:
			if isTrain==True: # 训练时显示进度，推荐时不需要
				faceCnt += 1
				print("%.2f%%" % (faceCnt*100/len(faceData)))
			dim2X = self.tsne.fit_transform(x)
			newX.append(dim2X.mean(axis=0))
		return newX

	def train(self):
		if len(self.y)<2:
			print("样本少于 2 个，请继续选择")
			return
		print("开始训练")

		# 先将 32*32 图形矩阵降维至 2*32，然后计算两列的平均值之后再分类
		newX = self.dimentionTransform(self.X,True)
		self.clsf.fit(newX,self.y)
		# print("训练完成，共 %d 个正例 %d 个反例。" % (self.y.count(1), self.y.count(0)))
		print("训练完成")

	def chkType(self,faceData):
		clsf = joblib.load('./train.mdl')
		newFaces = self.dimentionTransform(faceData)
		arr = clsf.predict(newFaces)
		score = clsf.predict_proba(newFaces) # 只有当使用决策树时才用，其他方式需要修改返回的 score 为 None
		return arr,score

	def saveModule(self):
		print("开始保存模型")
		joblib.dump(self.clsf,'./train.mdl')
		print("保存结束")
