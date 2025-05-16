
import math
import time
import cv2
import mediapipe as mp 


class PoseDetector:
    def __init__(self,mode=False,smooth=True,detectionCon = 0.5, trackCon=0.5):
        self.mode = mode 
        self.smooth=smooth
        self.detectionCon = detectionCon
        self.trackCon=trackCon
        self.mpPose = mp.solutions.pose 
        self.mpDraw = mp.solutions.drawing_utils
        self.pose = self.mpPose.Pose(
            static_image_mode=self.mode,
            smooth_landmarks=self.smooth,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon,
            model_complexity=1,
            enable_segmentation=False,
            smooth_segmentation=True)
      
    def findPose(self,img,draw=True):
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)    
        if self.results.pose_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img,self.results.pose_landmarks,self.mpPose.POSE_CONNECTIONS)
        return img
    
    def findPosition(self,img,draw = True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id,lm in enumerate(self.results.pose_landmarks.landmark):
                h,w,c  = img.shape 
                cx,cy = int(lm.x*w),int(lm.y*h)
                self.lmList.append([id,cx,cy])
                # cv2.circle(img,(cx,cy),5,(255,0,0),3)
        return self.lmList

    def findAngle(self,img,p1,p2,p3,draw=True):
        x1,y1 = self.lmList[p1][1:]
        x2,y2 = self.lmList[p2][1:]
        x3,y3 = self.lmList[p3][1:]
        
        # Calculate Angle
        angle = math.degrees(math.atan2(y3-y2,x3-x2) - math.atan2( y1-y2,x1-x2))
        if angle<0:
            angle+=360
        
        if draw: 
            cv2.line(img,(x1,y1),(x2,y2),(0,255,0),3)
            cv2.line(img,(x2,y2),(x3,y3),(0,255,0),3)
            cv2.circle(img, (x1,y1),10,(255,0,0),cv2.FILLED)
            cv2.circle(img, (x1,y1),15,(255,0,0),2)
            
            cv2.circle(img, (x2,y2),10,(255,0,0),cv2.FILLED)
            cv2.circle(img, (x2,y2),15,(255,0,0),2)
            
            cv2.circle(img, (x3,y3),10,(255,0,0),cv2.FILLED)
            cv2.circle(img, (x3,y3),15,(255,0,0),2)
            # cv2.putText(img, str(int(angle)),(x2-50,y2+50), cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
        return angle
        





def main():
    capture = cv2.VideoCapture("assets/dance.mp4") 

    pTime = 0
    detector = PoseDetector()

    while True: 
        success,img = capture.read()
        
        if success:
            img = detector.findPose(img)
            lmList = detector.findPosition(img)
            if len(lmList)!=0:
                print(lmList[14])   
                cv2.circle(img,(lmList[14][1],lmList[14][2]),5,(255,183,255),3)
       
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img,str(int(fps)),(70,50),cv2.FONT_HERSHEY_COMPLEX,3,(255,0,0),3)
        cv2.imshow("Pose Capture",img)
        cv2.waitKey(1)



if __name__ == "__main__":
     main()