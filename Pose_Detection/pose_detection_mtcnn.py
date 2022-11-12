from re import X
import numpy as np
import cv2
import csv
import os

## MTCNN source code is add to the mtcnn folder [if you want install : $ pip install mtcnn ] link :https://github.com/ipazc/mtcnn
from mtcnn.mtcnn import MTCNN


detector = MTCNN()

def detect_faces(image, image_shape_max=640):
    '''
    Parameters
    ----------
    image for face detection.
    image_shape_max : int, optional
        maximum size (in pixels) of image.

    Returns
    -------
    bounding boxes and score.
    landmarks.

    '''

    image_shape = image.shape[:2]
    
    # perform image resize for faster detection    
    if image_shape_max:
        scale_factor = max([1, max(image_shape) / image_shape_max])
    else:
        scale_factor = 1
        
    if scale_factor > 1:        
        scaled_image = cv2.resize(image, (0, 0), fx = 1/scale_factor, fy = 1/scale_factor)
        bbs, points = detector.detect_faces(scaled_image)
        bbs[:,:4] *= scale_factor
        points *= scale_factor
    else:
        bbs, points = detector.detect_faces(image)
    
    return bbs, points



def one_face(frame, bbs, pointss):
    """
    Parameters
    ----------
    frame : TYPE
        RGB image (numpy array).
    bbs : TYPE - Array of flaot64, Size = (N, 5)
        coordinates of bounding boxes for all detected faces.
    pointss : TYPE - Array of flaot32, Size = (N, 10)
        coordinates of landmarks for all detected faces.

    Returns
    -------
    bb : TYPE - Array of float 64, Size = (5,)
        coordinates of bounding box for the selected face.
    points : TYPE
        coordinates of five landmarks for the selected face.

    """
    # select only process only one face (center ?)
    offsets = [(bbs[:,0]+bbs[:,2])/2-frame.shape[1]/2,
               (bbs[:,1]+bbs[:,3])/2-frame.shape[0]/2]
    offset_dist = np.sum(np.abs(offsets),0)
    index = np.argmin(offset_dist)
    bb = bbs[index]
    points = pointss[:,index]
    return bb, points



def find_pose(points):
    """
    Parameters
    ----------
    points : TYPE - Array of float32, Size = (10,)
        coordinates of landmarks for the selected faces.
    Returns
    -------
    Angle
    Yaw
    Pitch
    TYPE
        pitch of face.

    """
    LMx = points[0:5]# horizontal coordinates of landmarks
    LMy = points[5:10]# vertical coordinates of landmarks
    
    dPx_eyes = max((LMx[1] - LMx[0]), 1)
    dPy_eyes = (LMy[1] - LMy[0])
    angle = np.arctan(dPy_eyes / dPx_eyes) # angle for rotation based on slope eyes
    
    alpha = np.cos(angle)
    beta = np.sin(angle)
    # rotated landmarks
    LMxr = (alpha * LMx + beta * LMy + (1 - alpha) * LMx[2] / 2 - beta * LMy[2] / 2) 
    LMyr = (-beta * LMx + alpha * LMy + beta * LMx[2] / 2 + (1 - alpha) * LMy[2] / 2)
    
    # average distance between eyes and mouth
    dXtot = (LMxr[1] - LMxr[0] + LMxr[4] - LMxr[3]) / 2
    dYtot = (LMyr[3] - LMyr[0] + LMyr[4] - LMyr[1]) / 2
    
    # average distance between nose and eyes
    dXnose = (LMxr[1] - LMxr[2] + LMxr[4] - LMxr[2]) / 2
    dYnose = (LMyr[3] - LMyr[2] + LMyr[4] - LMyr[2]) / 2
    
    # relative rotation 0 degree is frontal 90 degree is profile
    Xfrontal = (-90+90 / 0.5 * dXnose / dXtot) if dXtot != 0 else 0
    Yfrontal = (-90+90 / 0.5 * dYnose / dYtot) if dYtot != 0 else 0

    return angle * 180 / np.pi, Xfrontal, Yfrontal








def mtcnn_method(image):
## Read the image 
    img = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)

    try:
        #bounding_boxes, landmarks = detector.detect_faces(image_rgb)
        bounding_boxes, landmarks = detect_faces(img)
        bbs = bounding_boxes.copy()
        lmarks = landmarks.copy()
    except:
        print("Error: face detector error.")




    # if at least one face is detected
    if len(bounding_boxes) > 0:

        bb, lmarks_5 = one_face(img, bbs, lmarks)
        angle, Xfrontal, Yfrontal = find_pose(lmarks_5)
      
        roll = angle
        yaw = Xfrontal
        pitch = Yfrontal
        

    else:
        print('no face detected')

    
    return roll , pitch , yaw






### for single object 

input_path='../examples/pose_7.jpg'
roll , pitch , yaw = mtcnn_method(input_path)
print(f" Roll:{roll} \n Pitch: {pitch} \n Yaw:{yaw}")

## for all the images in examples folder 
examples=os.listdir('../examples')
results=[]
for image in examples:
    '''
    Output : CSV file 
    '''
    roll , pitch , yaw = mtcnn_method(f"../examples/{image}")
    results.append({'Image_name':image,'Roll':roll,'Pitch':pitch,'Yaw':yaw})


header_name=['Image_name','Roll','Pitch','Yaw']

with open('roll_pitch_yaw_results_mtcnn_method.csv','w') as csvfile:
    writer=csv.DictWriter(csvfile,fieldnames=header_name)
    writer.writeheader()
    writer.writerows(results)