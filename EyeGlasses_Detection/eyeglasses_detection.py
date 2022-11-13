import cv2,sys,os,time,dlib
import numpy as np
from faceBlendCommon import getLandmarks
import os 
import csv
FACE_DOWNSAMPLE_RATIO = 2
RESIZE_HEIGHT = 360

predictions2Label = {0:"No Glasses", 1:"With Glasses"}

def svmPredict(model, samples):
  return model.predict(samples)[1].ravel()

def prepareData(data):
  featureVectorLength = len(data[0])
  features = np.float32(data).reshape(-1,featureVectorLength)
  return features

def computeHOG(hog, data):
  hogData = []
  for image in data:
    hogFeatures = hog.compute(image)
    hogData.append(hogFeatures)

  return hogData




# Load face detection and pose estimation models.
modelPath = "models/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(modelPath)

# Initialize hog parameters
winSize = (96,32)
blockSize = (8,8)
blockStride = (8,8)
cellSize = (4,4)
nbins = 9
derivAperture = 0
winSigma = 4.0
histogramNormType = 1
L2HysThreshold =  2.0000000000000001e-01
gammaCorrection = 1
nlevels = 64

hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,
                        derivAperture,winSigma,histogramNormType,
                        L2HysThreshold,gammaCorrection,nlevels,1)

# We will load the model again and test the model
savedModel = cv2.ml.SVM_load("models/eyeGlassClassifierModel.yml")
# Start webcam
  
def glass_detection(image):
  img = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
        # Read frame
  frame = img
  height, width = frame.shape[:2]
  IMAGE_RESIZE = np.float32(height)/RESIZE_HEIGHT
  frame = cv2.resize(frame,None,
                      fx=1.0/IMAGE_RESIZE,
                      fy=1.0/IMAGE_RESIZE,
                      interpolation = cv2.INTER_LINEAR)

  landmarks = getLandmarks(detector, predictor, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), FACE_DOWNSAMPLE_RATIO)

  x1 = landmarks[0][0]
  x2 = landmarks[16][0]
  y1 = min(landmarks[24][1], landmarks[19][1])
  y2 = landmarks[29][1]

  cropped = frame[y1:y2,x1:x2,:]
  cropped = cv2.resize(cropped,(96, 32), interpolation = cv2.INTER_CUBIC)

  testHOG = computeHOG(hog, np.array([cropped]))
  testFeatures = prepareData(testHOG)
  predictions = svmPredict(savedModel, testFeatures)
  frameClone = np.copy(frame)
  #cv2.putText(frameClone, "Prediction = {}".format(predictions2Label[int(predictions[0])]), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
  cv2.putText(frameClone, "Prediction = {}".format(predictions2Label[int(predictions[0])]), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0,0), 4)

  print("Prediction = {}".format(predictions2Label[int(predictions[0])]))

  return predictions2Label[int(predictions[0])]



input_path='../examples/pose_9.jpg'
glass_detection(input_path)


## for all the images in examples folder 
examples=os.listdir('../examples')
results=[]
for image in examples:
    '''
    Output : CSV file 
    '''
    try:
      results.append({'Image_name':image,'Result':glass_detection(f"../examples/{image}")})
    except:
      results.append({'Image_name':image,'Result':'Face Did not detect(check the pose)'})



header_name=['Image_name','Result']

with open('glass_detection_results.csv','w') as csvfile:
    writer=csv.DictWriter(csvfile,fieldnames=header_name)
    writer.writeheader()
    writer.writerows(results)