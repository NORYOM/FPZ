from tkinter import *
import tkinter.filedialog
import os
from PIL import Image,ImageTk
import faceDetector as fd
import cv2
from classify import Classify 

class UI:
	win = Tk()
	cv = Canvas(win,width=800,height=600,bg='gray')
	print("正在载入，请稍候。")

	imgPath = "" # 存放供选脸图片的文件夹路径
	imgNames = [] # 存放文件夹内所有图片文件名
	inputImg = None
	picCount = 0

	btnNextImg = None
	btnPrevImg = None

	pic = None
	colrs = ["orange","lightgreen"]
	colr = colrs[0]
	faceData = [] # 脸的位置尺寸等信息
	trainningData = [] # 灰度并缩放后的用于训练的脸
	typeSet = [] # 脸类型, 1: 感兴趣; 0: 不感兴趣

	def __init__(self):
		self.cv.bind("<Button-1>",self.mouseDownOnCanvas)

		btnSelectFolder = Button(self.win, text="选择图片文件夹", command=self.btnSelFoldClick)
		btnSelectFolder.grid(row=0,column=0,sticky="nw")
		self.btnNextImg = Button(self.win, text="下一张", command=self.btnNextImgClick)
		self.btnNextImg.grid(row=0,column=1,sticky="nw")
		self.btnPrevImg = Button(self.win, text="上一张", command=self.btnPrevImgClick)
		self.btnPrevImg.grid(row=0,column=2,sticky="nw")
		self.btnTrainImg = Button(self.win, text="训练并保存", command=self.btnSaveTrainDataClick)
		self.btnTrainImg.grid(row=0,column=3,sticky="nw")
		self.win.rowconfigure(0, weight=1)
		self.win.columnconfigure(3, weight=1)

		self.btnNextImg['state']="disabled"
		self.btnPrevImg['state']="disabled"

	def loadImgs(self):
		res = os.walk(self.imgPath)
		for root,dirs,files in res:
			for file in files:
				self.imgNames.append(self.imgPath+"/"+file)

	def showImgOnCanvas(self, imgName):
		self.cv.grid(row=1,column=0,columnspan=4)
		self.cv.delete(ALL)
		loadedImg = Image.open(imgName)
		pic = ImageTk.PhotoImage(loadedImg)
		self.inputImg = cv2.imread(imgName,cv2.IMREAD_COLOR)

		# 缩放
		h, w = self.inputImg.shape[:2]
		scale = 1
		if h>600 or w>800:
			scale = 600/max(h,w)
		dims = (int(w * scale), int(h * scale))

		# 缩放图片用于 tkinter 显示
		resizedPic = loadedImg.resize((dims[0],dims[1]))
		self.pic = ImageTk.PhotoImage(resizedPic)
		# 缩放图片用于 opencv 处理
		interpln = cv2.INTER_LINEAR if scale > 1.0 else cv2.INTER_AREA
		self.inputImg = cv2.resize(self.inputImg, dims, interpolation=interpln)

		self.drawPic()
		faces = self.getFaces(self.inputImg)
		self.drawFacePos(faces,self.colr)


	def drawPic(self):
		self.cv.create_image(0,0,anchor="nw",image=self.pic)

	def getFaces(self,inputImg):
		faces = fd.detect_faces_dnn(inputImg)
		return faces

	def drawFacePos(self,faces,colr):
		for face in faces:
			x = face[0]
			y = face[1]
			w = face[2]
			h = face[3]

			# 绘制脸部外框
			boadw = 20

			l11 = self.cv.create_line(x,y, x+boadw,y, fill=colr)
			l12 = self.cv.create_line(x,y, x,y+boadw, fill=colr)

			l21 = self.cv.create_line(x+w,y, x+w-boadw,y, fill=colr)
			l22 = self.cv.create_line(x+w,y, x+w,y+boadw, fill=colr)

			l31 = self.cv.create_line(x,y+h, x+boadw,y+h, fill=colr)
			l32 = self.cv.create_line(x,y+h, x,y+h-boadw, fill=colr)

			l41 = self.cv.create_line(x+w,y+h, x+w-boadw,y+h, fill=colr)
			l42 = self.cv.create_line(x+w,y+h, x+w,y+h-boadw, fill=colr)

			faceId = str(face[0]) + ":" + str(face[1]) + ":" + str(face[2]) + ":" + str(face[3])

			trainPic = self.processedPic(face,self.inputImg)

			faceInfo = {
				"imgNo.":self.picCount, # 当前处理的是文件夹中第几张图片
				"faceId":faceId, #标记唯一的脸
				"posSize":face,	#脸的尺寸位置
				"board":[l11,l12,l21,l22,l31,l32,l41,l42], #脸的线框
				"select":False, #是否被选中
				"trainPic": trainPic
			}

			# 检查是否已经加载过，保证没有重复的脸数据
			faceExisted = False
			for face in self.faceData:
				if face != None and face["faceId"]==faceId:
					faceExisted = True
					break

			# 如果已经有这个脸，继续处理后面的脸
			if faceExisted==True:
				face["board"] = faceInfo["board"]
				self.setFaceSelect(face)
				continue
			# 如果不存在重复的脸，加入这个脸
			self.faceData.append(faceInfo)

	def isInFace(self,mx,my,face):
		x = face[0]
		y = face[1]
		w = face[2]
		h = face[3]
		if mx>=x and mx<=x+w and my>=y and my<=y+h:
			return True
		return False

	def processedPic(self,face,inputImg): # 灰度化脸并缩放到 32*32
		x = face[0]
		y = face[1]
		w = face[2]
		h = face[3]
		img = inputImg[y:y+h,x:x+w]
		faceImg = img.copy()
		grayFace = cv2.cvtColor(faceImg,cv2.COLOR_BGR2GRAY)
		scaledFace = cv2.resize(grayFace,(32,32))
		return scaledFace

	def setFaceSelect(self,face):
		if face["select"]==True:
			colr = self.colrs[1]
		else:
			colr = self.colrs[0]
		self.cv.itemconfig(face["board"][0], fill=colr)
		self.cv.itemconfig(face["board"][1], fill=colr)
		self.cv.itemconfig(face["board"][2], fill=colr)
		self.cv.itemconfig(face["board"][3], fill=colr)
		self.cv.itemconfig(face["board"][4], fill=colr)
		self.cv.itemconfig(face["board"][5], fill=colr)
		self.cv.itemconfig(face["board"][6], fill=colr)
		self.cv.itemconfig(face["board"][7], fill=colr)

	def mouseDownOnCanvas(self,e):
		for face in self.faceData:
			# 加入 face["imgNo."]==self.picCount 是为了判断并排除不同图片里同一个坐标的不同的脸
			if self.isInFace(e.x,e.y,face["posSize"]) and face["imgNo."]==self.picCount:
				face["select"] = not face["select"]
				self.setFaceSelect(face)

	def run(self):
		self.win.mainloop()

	def btnSelFoldClick(self):
		self.imgPath = tkinter.filedialog.askdirectory()
		if len(self.imgPath)>0:
			self.imgNames = []
			self.picCount = 0
			self.faceData = [] # 重新选择文件夹就重置
			self.loadImgs()
			self.btnNextImg['state']="active"
			self.showImgOnCanvas(self.imgNames[self.picCount])

	def btnNextImgClick(self):
		self.picCount += 1
		self.showImgOnCanvas(self.imgNames[self.picCount])
		if self.picCount>=len(self.imgNames)-1:
			self.btnNextImg['state']="disabled"
			return
		if self.picCount>0 and len(self.imgNames)>1:
			self.btnPrevImg['state']="active"

	def btnPrevImgClick(self):
		self.picCount -= 1
		self.showImgOnCanvas(self.imgNames[self.picCount])
		if self.picCount<=0:
			self.btnPrevImg['state']="disabled"
			return
		if self.picCount>0 and self.picCount<=len(self.imgNames)-1:
			self.btnNextImg['state']="active"

	def btnSaveTrainDataClick(self):
		self.trainningData = [] # 清空训练数据
		self.typeSet = [] # 清空类型数据
		for face in self.faceData:
			self.trainningData.append(face["trainPic"])
			if face["select"]==True:
				self.typeSet.append(1)
			else:
				self.typeSet.append(0)
		clf = Classify()
		clf.setTrainData(self.trainningData)
		clf.setTypeData(self.typeSet)
		clf.train()
		clf.saveModule()

ui = UI()
ui.run()
