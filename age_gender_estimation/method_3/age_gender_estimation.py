from deepface import DeepFace
import os 
import csv
examples=os.listdir('./age_gender_examples')
results=[]
for image in examples:
    '''
    Output : CSV file 
    '''
    try:
      obj = DeepFace.analyze(img_path = f"./age_gender_examples/{image}", actions = ['age', 'gender'])
      results.append({'Image_name':image,'Gender':obj['gender'],'Age':obj['age']})
    except:
      results.append({'Image_name':image,'Gender':'not','Age':'not'})



header_name=['Image_name','Gender','Age']

with open('age_gender_estimation_results.csv','w') as csvfile:
    writer=csv.DictWriter(csvfile,fieldnames=header_name)
    writer.writeheader()
    writer.writerows(results)