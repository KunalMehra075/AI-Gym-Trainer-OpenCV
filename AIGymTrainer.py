import cv2 
import numpy as np  
import time   
from poseEstimationModule import PoseDetector

capture = cv2.VideoCapture("assets/gym.mp4")

detector = PoseDetector()

direction = 0
count = 0
pTime = 0

while True :
    success,img = capture.read()
    # img = cv2.imread("assets/trainer/trainimage.jpg")
    img = detector.findPose(img,False)
    lmList = detector.findPosition(img,False)
    
    if len(lmList)!=0:
        # left arm
        angle = detector.findAngle(img,12,14,16)
        per = int(np.interp(angle,(42,178),(0,100)))
        # print(angle,per)
        
        
        if 90 <= per <= 100 :
            if direction == 0:
                count += 0.5
                direction = 1
        if 0 <= per <= 5:
            if direction == 1:
                  count += 0.5
                  direction = 0

  
        cv2.rectangle(img,(30,400),(120,465),(0,255,0),cv2.FILLED)
        cv2.putText(img,f"{count}",(35,450),cv2.FONT_HERSHEY_PLAIN,3,(233,0,0),3)
        
      
                
                
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img,f"FPS:{int(fps)}",(70,50),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)

    cv2.imshow("AI Trainer",img)
    cv2.waitKey(1)
    