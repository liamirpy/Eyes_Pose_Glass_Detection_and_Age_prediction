from pathlib import Path
from re import A
from xml.etree.ElementPath import prepare_predicate
import cv2
import dlib
import numpy as np
import argparse
from contextlib import contextmanager
from omegaconf import OmegaConf
from tensorflow.keras.utils import get_file
from src.factory import get_model

import os 
import csv

margin=0.4

weight_file='/Users/amir/Desktop/age_gneder/age-gender-estimation/pretrained_models/EfficientNetB3_224_weights.11-3.44.hdf5'


def yield_images_from_dir(image_dir):
    image_dir = Path(image_dir)

    img = cv2.imread(str(image_dir), 1)

    if img is not None:
        h, w, _ = img.shape
        r = 640 / max(w, h)
        yield cv2.resize(img, (int(w * r), int(h * r)))



def gender_age_estimation(image_dir):
    

    # for face detection
    detector = dlib.get_frontal_face_detector()

    # load model and weights
    model_name, img_size = Path(weight_file).stem.split("_")[:2]
    img_size = int(img_size)
    cfg = OmegaConf.from_dotlist([f"model.model_name={model_name}", f"model.img_size={img_size}"])
    model = get_model(cfg)
    model.load_weights(weight_file)

    image_generator = yield_images_from_dir(image_dir) 

    for img in image_generator:
        
        input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = np.shape(input_img)

        # detect faces using dlib detector
        detected = detector(input_img,1)

        faces = np.empty((len(detected), img_size, img_size, 3))
        if len(detected) > 0:
            for i, d in enumerate(detected):
                x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
                xw1 = max(int(x1 - margin * w), 0)
                yw1 = max(int(y1 - margin * h), 0)
                xw2 = min(int(x2 + margin * w), img_w - 1)
                yw2 = min(int(y2 + margin * h), img_h - 1)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                # cv2.rectangle(img, (xw1, yw1), (xw2, yw2), (255, 0, 0), 2)
                faces[i] = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1], (img_size, img_size))

            # predict ages and genders of the detected faces
            results = model.predict(faces)
            predicted_genders = results[0]
            ages = np.arange(0, 101).reshape(101, 1)
            predicted_ages = results[1].dot(ages).flatten()

            # draw results
            for i, d in enumerate(detected):
                label = "{}, {}".format(int(predicted_ages[i]),"M" if predicted_genders[i][0] < 0.5 else "F")
                age=int(predicted_ages[i])
                gender="M" if predicted_genders[i][0] < 0.5 else "F"
            




        return age,gender
        




examples=os.listdir('./age_gender_examples')
results=[]
for image in examples:
    '''
    Output : CSV file 
    '''
    age , gender  = gender_age_estimation(f"./age_gender_examples/{image}")
    results.append({'Image_name':image,'Gender':gender,'Age':age})


header_name=['Image_name','Gender','Age']

with open('age_gender_estimation_results.csv','w') as csvfile:
    writer=csv.DictWriter(csvfile,fieldnames=header_name)
    writer.writeheader()
    writer.writerows(results)
