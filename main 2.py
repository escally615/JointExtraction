import mediapipe as mp
import cv2

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

cap = cv2.VideoCapture(0)

poseAcc = 0.0
totAcc = 0.0
avgAcc = 0.0

#correct={}
#correct[0] = (305,101)
#correct[1] =  (309, 96)
#correct[2]= (309, 96)
#correct[3] = (309, 96)
#correct[4] = (311, 95)
#correct[5] = (312, 95)
#correct[6] = (314, 95)
#correct[7] = (316, 100)
#correct[8] = (322, 99)
#correct[9] = (307, 107)
#correct[10] = (309, 107)
#correct[11] = (302, 141)
#correct[12] = (344, 139)
#correct[13] = (258, 147)
#correct[14] = (386, 144)
#correct[15] = (251, 142)
#correct[16] = (420, 161)
#correct[17] = (247, 144)
#correct[18] = (431, 165)
#correct[19] = (247, 142)
#correct[20] = (431, 165)
#correct[21] = (250, 142)
#correct[22] = (426, 164)
#correct[23] = (349, 234)
#correct[24] = (366, 223)
#correct[25] = (365, 310)
#correct[26] = (412, 248)
#correct[27] = (372, 389)
#correct[28] = (467, 261)
#correct[29] = (376, 399)
#correct[30] = (473, 263)
#correct[31] = (337, 408)
#correct[32] = (482, 264)


while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        points = {}
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h,w,c = img.shape
            cx, cy = int(lm.x*w), int(lm.y*h)
            print(id, cx, cy)
            #points[id] = (cx, cy)
            #poseAcc = abs(((((points[id][0]-correct[id][0])/correct[id][0])+((points[id][1]-correct[id][1])/correct[id][1]))/2)*100)
            #totAcc+=poseAcc

        #avgAcc = totAcc/32
        #totAcc = 0



    cv2.putText(img, (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0 , 0), 2)






    cv2.imshow("Arabesque Pose Estimator", img)
    cv2.waitKey(1)
