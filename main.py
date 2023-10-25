import cv2
import numpy as np

# ############################
# display image

img = cv2.imread("car.png")
cv2.imshow("Image",img)

wht = 320
conft = 0.5
nmst= 0.2

################################33
def findObjects(outputs,img):
    hT,wT,cT = img.shape
    bbox = []
    classID = []
    confs = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId =np.argmax(scores)
            conf = scores[classId]
            if conf > conft:
                w,h = int(det[2]*wT),int(det[3]*hT)
                x,y = int((det[0]*wT)-w/2) , int((det[1]*hT)-hT/2)
                bbox.append([x,y,w,h])
                classID.append(classId)
                confs.append(float(conf))
    indices = cv2.dnn.NMSBoxes(bbox,confs,conft,nmst)
    for i in indices:
       box = bbox[i]
       x,y,w,h = box[0],box[1],box[2],box[3]
       cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)



# YOLO is trained on COCO datasets

classfile = 'coco.names'
classname = []
with open(classfile,'rt') as f:
    classname = f.read().rstrip('\n').rsplit('\n')

modelconf = 'yolov3-tiny.cfg'
modelweig = 'yolov3-tiny.weights'

net = cv2.dnn.readNetFromDarknet(modelconf,modelweig)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
blob = cv2.dnn.blobFromImage(img,1/255,(wht,wht),[0,0,0],1,crop=False)
net.setInput(blob)
layerNames = net.getLayerNames()

outputnames = [layerNames[i-1] for i in net.getUnconnectedOutLayers()]

outputs = net.forward(outputnames)
findObjects(outputs,img)



cv2.waitKey(0)