from asyncore import write
from cmath import pi
from tkinter import Y
from unittest import result
import numpy as np
import cv2
import tensorflow as tf
import os 
import csv 

# load retinaFace face detector
detector_model = tf.saved_model.load('./models/tf_retinaface_mbv2/')


def one_face(frame, bbs, pointss):
    """
    Parameters
    ----------
    frame : uint8
        RGB image (numpy array).
    bbs : float64, Size = (N, 4)
        coordinates of bounding boxes for all detected faces.
    pointss : flaot32, Size = (N, 10)
        coordinates of landmarks for all detected faces.

    Returns
    -------
    bb : float64, Size = (5,)
        coordinates of bounding box for the selected face.
    points : float32
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
    points : float32, Size = (10,)
        coordinates of landmarks for the selected faces.
    Returns
    -------
    float32, float32, float32
    """
    LMx = points[0:5]# horizontal coordinates of landmarks
    LMy = points[5:10]# vertical coordinates of landmarks
    
    dPx_eyes = max((LMx[1] - LMx[0]), 1)
    dPy_eyes = (LMy[1] - LMy[0])
    angle = np.arctan(dPy_eyes / dPx_eyes) # angle for rotation based on slope
    
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



def recover_pad_output(outputs, pad_params):
    """recover the padded output effect"""
    img_h, img_w, img_pad_h, img_pad_w = pad_params
    recover_xy = np.reshape(outputs[:, :14], [-1, 7, 2]) * \
        [(img_pad_w + img_w) / img_w, (img_pad_h + img_h) / img_h]
    outputs[:, :14] = np.reshape(recover_xy, [-1, 14])

    return outputs


def pad_input_image(img, max_steps=32):
    """pad image to suitable shape - required for retinaface"""
    img_h, img_w, _ = img.shape

    img_pad_h = 0
    if img_h % max_steps > 0:
        img_pad_h = max_steps - img_h % max_steps

    img_pad_w = 0
    if img_w % max_steps > 0:
        img_pad_w = max_steps - img_w % max_steps

    padd_val = np.mean(img, axis=(0, 1)).astype(np.uint8)
    img = cv2.copyMakeBorder(img, 0, img_pad_h, 0, img_pad_w,
                             cv2.BORDER_CONSTANT, value=padd_val.tolist())
    pad_params = (img_h, img_w, img_pad_h, img_pad_w)

    return img, pad_params


def retinaface(image):
    """ retinaface face detector"""

    height = image.shape[0]
    width = image.shape[1]
    
    image_pad, pad_params = pad_input_image(image)    
    image_pad = tf.convert_to_tensor(image_pad[np.newaxis, ...])
    image_pad = tf.cast(image_pad, tf.float32)  
   
    outputs = detector_model(image_pad).numpy()

    outputs = recover_pad_output(outputs, pad_params)
    Nfaces = len(outputs)
    
    bbs = np.zeros((Nfaces,5))
    lms = np.zeros((Nfaces,10))
    
    bbs[:,[0,2]] = outputs[:,[0,2]]*width
    bbs[:,[1,3]] = outputs[:,[1,3]]*height
    bbs[:,4] = outputs[:,-1]
    
    lms[:,0:5] = outputs[:,[4,6,8,10,12]]*width
    lms[:,5:10] = outputs[:,[5,7,9,11,13]]*height
    
    return bbs, lms



def detect_faces(image, image_shape_max=640):
    '''
    Performs face detection using retinaface method with speed boost and 
    initial quality checks based on whole image size
    
    Parameters
    ----------
    image : uint8
        image for face detection.
    image_shape_max : int, optional
        maximum size (in pixels) of image.

    Returns
    -------
    float array
        landmarks.
    float array
        bounding boxes.
    flaot array
        detection scores.
    '''

    image_shape = image.shape[:2]
    
    # perform image resize for faster detection    
    if image_shape_max:
        scale_factor = max([1, max(image_shape)/image_shape_max])
    else:
        scale_factor = 1
        
    if scale_factor > 1:        
        scaled_image = cv2.resize(image, (0, 0), fx = 1 / scale_factor, 
                                  fy = 1 / scale_factor)
        bbs_all, points_all = retinaface(scaled_image)
        bbs_all[:,:4] *= scale_factor
        points_all *= scale_factor
    else:
        bbs_all, points_all = retinaface(image)              
    
    scores = bbs_all[:,-1]
    bbs = bbs_all[:, :4]
    
    return points_all, bbs, scores





def retinaface_method(image):

    img = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)



    try:
        landmarks, bboxes, scores = detect_faces(img,image_shape_max=640)
    except:
        print("Error: face detector error.")




    if len(bboxes) > 0:
            
        #bboxes = np.insert(bboxes,bboxes.shape[1], scores, axis=1)
        lmarks = np.transpose(landmarks)
        bbs = bboxes.copy()

        # process only one face (center ?) if multiple faces detected  
        bb, lmarks_5 = one_face(img, bbs, lmarks)
        
        
        angle, Xfrontal, Yfrontal = find_pose(lmarks_5)
        
        
        roll = angle
        yaw = Xfrontal
        pitch = Yfrontal
    

    return roll , pitch , yaw

## 4 6 7 13
## 7 is problem 


### for single object 

input_path='../examples/pose_7.jpg'
roll , pitch , yaw = retinaface_method(input_path)
print(f" Roll:{roll} \n Pitch: {pitch} \n Yaw:{yaw}")

## for all the images in examples folder 
examples=os.listdir('../examples')
results=[]
for image in examples:
    '''
    Output : CSV file 
    '''
    roll , pitch , yaw = retinaface_method(f"../examples/{image}")
    results.append({'Image_name':image,'Roll':roll,'Pitch':pitch,'Yaw':yaw})


header_name=['Image_name','Roll','Pitch','Yaw']

with open('roll_pitch_yaw_results_retinaface_method.csv','w') as csvfile:
    writer=csv.DictWriter(csvfile,fieldnames=header_name)
    writer.writeheader()
    writer.writerows(results)