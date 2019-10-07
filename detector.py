import cv2
import time

############### if python version is <=2.7 comment following
# detector = cv2.dnn.readNetFromCaffe("resnet_ssd_v1.prototxt","resnet_ssd_v1.caffemodel")
# detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
############### if python version is <=2.7 comment above
target = (300, 300)
inputImg = cv2.imread('01.png',cv2.IMREAD_COLOR)
configConfidence = 50/100

face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
face_lbp = cv2.CascadeClassifier('./lbpcascade_frontalface_improved.xml')

def compile_detection_image(input_image):  # pylint:disable=too-many-arguments
    """ Compile the detection image """
    image = input_image.copy()
    scale = set_scale(image)
    image = scale_image(image, scale)
    return [image, scale]

def set_scale(image):
    """ Set the scale factor for incoming image """
    height, width = image.shape[:2]

    # if isinstance(target, int):
    #     dims = (target ** 0.5, self.target ** 0.5)
    #     target = dims
    source = max(height, width)

    scale = target[0] / source

    return scale

def scale_image(image, scale):
    """ Scale the image and optional pad to given size """
    # pylint: disable=no-member
    height, width = image.shape[:2]
    interpln = cv2.INTER_LINEAR if scale > 1.0 else cv2.INTER_AREA
    if scale != 1.0:
        dims = (int(width * scale), int(height * scale))
    #     if scale < 1.0:
    #         print("Resizing image from %sx%s to %s. Scale=%s" 
				# %(width, height, "x".join(str(i) for i in dims), scale))
        image = cv2.resize(image, dims, interpolation=interpln)
    return image

def to_bounding_box_dict(left, top, right, bottom):
    """ Return a dict for the bounding box """
    return [int(round(left)),
    		int(round(top)),
			int(round(right))-int(round(left)),
			int(round(bottom))-int(round(top))]

def process_output(faces, scale, width, height):
    """ Compile found faces for output """
    # face[0] -- left
    # face[1] -- top
    # face[2] -- right
    # face[3] -- bottom

    faces = [to_bounding_box_dict(face[0] / scale, face[1] / scale, face[2] / scale, face[3]/ scale) for face in faces]

    # print("Processed Output: %s" %faces)
    return faces

def detect_faces_dnn(inputImg):

	[detect_image, scale] = compile_detection_image(inputImg) 
	height, width = detect_image.shape[:2]
	for angle in [0]:
		current_image = detect_image
		# print("Detecting faces")
		blob = cv2.dnn.blobFromImage(current_image,
										1.0,target,
										[104, 117, 123],
										False,
										False)
		detector.setInput(blob)
		detected = detector.forward()
		faces = list()
		for i in range(detected.shape[2]):
			confidence = detected[0,0,i,2]
			if confidence >= configConfidence:
				# print("Accepting due to confidence %s >= %s" %(confidence, configConfidence)) 
				faces.append([(detected[0, 0, i, 3] * width),
							(detected[0, 0, i, 4] * height),
							(detected[0, 0, i, 5] * width),
							(detected[0, 0, i, 6] * height)])

		# print("Detected faces: %s" %([face for face in faces]))

		return process_output(faces, scale, width, height)

def detect_faces_cascad(inputImg):
	faces = face_cascade.detectMultiScale(inputImg)
	return faces


def detect_faces_lbp(inputImg):
	faces = face_lbp.detectMultiScale(inputImg)
	return faces

def drawPos(pic,x,y,w,h,txt,colr):
	boadw = 10
	cv2.line(pic,(x,y),(x+boadw,y),colr,1)
	cv2.line(pic,(x,y),(x,y+boadw),colr,1)

	cv2.line(pic,(x+w,y),(x+w-boadw,y),colr,1)
	cv2.line(pic,(x+w,y),(x+w,y+boadw),colr,1)

	cv2.line(pic,(x,y+h),(x+boadw,y+h),colr,1)
	cv2.line(pic,(x,y+h),(x,y+h-boadw),colr,1)

	cv2.line(pic,(x+w,y+h),(x+w-boadw,y+h),colr,1)
	cv2.line(pic,(x+w,y+h),(x+w,y+h-boadw),colr,1)

	font = cv2.FONT_HERSHEY_SIMPLEX
	cv2.putText(pic,txt,(x,y-10), font, 0.5,colr,1,cv2.LINE_AA)


print("start detect")

############### if python version is <=2.7 comment following
# start_dnn = time.time()
# faces_dnn = detect_faces_dnn(inputImg)
# end_dnn = time.time()
############### if python version is <=2.7 comment above

start_cascad = time.time()
faces_cascad = detect_faces_cascad(inputImg)
end_cascad = time.time()

start_lbp = time.time()
faces_lbp = detect_faces_lbp(inputImg)
end_lbp = time.time()

############### if python version is <=2.7 comment following
# print("dnn : {:.5} sec ".format(end_dnn-start_dnn))
# for pos_dnn in faces_dnn:
# 	drawPos(inputImg,pos_dnn[0],pos_dnn[1],pos_dnn[2],pos_dnn[3],"DNN",(0,128,225))
############### if python version is <=2.7 comment above
print("cascad : {:.5} sec   lbp : {:.5} sec".format(end_cascad-start_cascad, end_lbp-start_lbp))
for pos_cascad in faces_cascad:
	drawPos(inputImg,pos_cascad[0],pos_cascad[1],pos_cascad[2],pos_cascad[3],"CASCAD",(0,200,225))
for pos_lbp in faces_lbp:
	drawPos(inputImg,pos_lbp[0],pos_lbp[1],pos_lbp[2],pos_lbp[3],"LBP",(0,225,128))

print(" press any key close")
cv2.imshow('image',inputImg)
cv2.waitKey(0)
cv2.destroyAllWindows()