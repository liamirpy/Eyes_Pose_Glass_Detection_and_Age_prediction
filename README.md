
####### 

Pose_detection 

Two methods are developed for pose detection, and the results for the images in example folder are added as CSV files.
Glasses detection

For eyeglasses, the results are added too.

This repository contains two parts:
1-Pose detection
2-Eyeglasses detection 
The result is added to each file.
All the results are in the all_result folder.

### Suggestion
It is better to first calculate the Pose degrees for each face and then check whether the person wears the glasses or not
### Problem 
The glasses detection method cannot detect the glasses in Three situations.
1-	The pose face is more than almost 45 degrees. 
2-	The skin color is too dark, and wearing the sunglass simultaneously(glasses_6 and glasses_7 in examples folder ) 

3-	Wearing huge eyes glass
### solution
1-	Before applying the glasses detection model, check the face pose.
2-	
3-	Celeba-hq mask, include the skin and glasses as an annotation. 

