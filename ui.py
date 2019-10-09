from tkinter import *
from PIL import Image,ImageTk
import faceDetector as fd
import cv2

class UI:
	win = Tk()
	cv = Canvas(win,width=800,height=600,bg='gray')
	imgName = 't1.jpg'
	loadedImg = Image.open(imgName)
	pic = ImageTk.PhotoImage(loadedImg)
	inputImg = cv2.imread(imgName,cv2.IMREAD_COLOR)

	# scale
	h, w = inputImg.shape[:2]
	print(max(h, w))
	scale = 600/max(h,w)
	dims = (int(w * scale), int(h * scale))

	# scale for tkinter canvas
	resizedPic = loadedImg.resize((dims[0],dims[1]))
	pic = ImageTk.PhotoImage(resizedPic)
	# scale for opencv
	interpln = cv2.INTER_LINEAR if scale > 1.0 else cv2.INTER_AREA
	inputImg = cv2.resize(inputImg, dims, interpolation=interpln)

	colrs = ["orange","lightgreen"]
	colr = colrs[0]
	faceData = []

	def __init__(self):
		self.cv.pack()
		self.cv.delete(ALL)

		self.drawPic()
		faces = self.getFaces(self.inputImg)
		self.drawFacePos(faces,self.colr)

		self.cv.bind("<Button-1>",self.mouseDown)

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

			boadw = 20

			l11 = self.cv.create_line(x,y, x+boadw,y, fill=colr)
			l12 = self.cv.create_line(x,y, x,y+boadw, fill=colr)

			l21 = self.cv.create_line(x+w,y, x+w-boadw,y, fill=colr)
			l22 = self.cv.create_line(x+w,y, x+w,y+boadw, fill=colr)

			l31 = self.cv.create_line(x,y+h, x+boadw,y+h, fill=colr)
			l32 = self.cv.create_line(x,y+h, x,y+h-boadw, fill=colr)

			l41 = self.cv.create_line(x+w,y+h, x+w-boadw,y+h, fill=colr)
			l42 = self.cv.create_line(x+w,y+h, x+w,y+h-boadw, fill=colr)

			self.faceData.append([face,l11,l12,l21,l22,l31,l32,l41,l42,0])

	def isInFace(self,mx,my,face):
		x = face[0]
		y = face[1]
		w = face[2]
		h = face[3]
		if mx>=x and mx<=x+w and my>=y and my<=y+h:
			return True
		return False

	def mouseDown(self,e):
		for data in self.faceData:
			if self.isInFace(e.x,e.y,data[0]):
				data[9] += 1
				colr = self.colrs[data[9]%2]
				self.cv.itemconfig(data[1], fill=colr)
				self.cv.itemconfig(data[2], fill=colr)
				self.cv.itemconfig(data[3], fill=colr)
				self.cv.itemconfig(data[4], fill=colr)
				self.cv.itemconfig(data[5], fill=colr)
				self.cv.itemconfig(data[6], fill=colr)
				self.cv.itemconfig(data[7], fill=colr)
				self.cv.itemconfig(data[8], fill=colr)

	def run(self):
		self.win.mainloop()

ui = UI()
ui.run()