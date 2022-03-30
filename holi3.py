from cmath import pi
import mediapipe as mp
import cv2
import math
import requests
import queue
import socket
import time
import json
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_pose = mp.solutions.pose

LEARNING_TIME = 3
WAITING_TIME = 3
WAITKEY = 100
DRIVER_ID = "001"
#ImageRecord(DRIVER_ID,image,lefthand,righthand,statusH,statusV,timestringserver, time.time())
class ImageRecord:
    def __init__(self,DRIVER_ID,image,lefthand,righthand,statusH,statusV,timestringserver, timeNow, message = ""):
        self.DRIVER_ID = DRIVER_ID
        self.image = image
        self.lefthand = lefthand
        self.righthand = righthand
        self.statusH = statusH
        self.statusV = statusV
        self.timestringserver = timestringserver
        self.time = timeNow
        self.message = message

    def statusReturn(self):
        data = {"DRIVER_ID" : self.DRIVER_ID, "lefthand": self.lefthand, "righthand": self.righthand, "statusH": self.statusH, "statusV": self.statusV, "timestringserver":self.timestringserver, "time":self.time, "message":self.message}
        #jsonData = json.dumps(data)
        return data

    def imageReturn(self):
        return self.image
    
    def updateMessage(self, message):
        self.message = message

    def driverReturn(self):
        return self.DRIVER_ID
    

def twopointdis(p1, p2):
    return math.sqrt(pow(p1[0]-p2[0],2) + pow(p1[1]-p2[1],2))


def threepoint(p1,p2,mid):
    left2mid = 0
    right2mid = 0
    leftside = twopointdis(p1,mid)
    rightside = twopointdis(p2,mid)
    longside = twopointdis(p1,p2)
    leftcos = (pow(leftside,2) + pow(longside,2)- pow(rightside,2))/(2*leftside*longside)
    rightcos = (pow(rightside,2) + pow(longside,2)- pow(leftside,2) )/(2*rightside*longside)
    longcos = (pow(leftside,2) + pow(rightside,2)- pow(longside,2))/(2*leftside*rightside)
    longcos = 180/pi*math.acos(longcos)
    #print(leftcos,' : ',rightcos,' : ',longcos,' : ',leftcos+rightcos+longcos)
    left2mid = leftside * leftcos
    right2mid2 = rightside * rightcos
    right2mid = round(longside - left2mid,10)
    left2mid = round(leftside * leftcos,10)
    #print(leftside,' : ',rightside)
    #print(longside,' : ',left2mid)
    #print(right2mid,' : ',right2mid2)
    return left2mid , right2mid

def facehoridir(leftobj,rightobj,nose_coor,image,mode='difference',standard = 0.5,compare=False):
    leftdiff, rightdiff = threepoint(leftobj,rightobj,nose_coor)
    
    roundleftdiff = round(leftdiff,5)
    roundrightdiff = round(rightdiff,5)
    status = "mid"
    if mode == 'ratio':
        if leftdiff > rightdiff:
            ratio = rightdiff/leftdiff
            if rightdiff/leftdiff >0.5:
                
                textdirection =  "direct Mid: l: {}, r: {}, ratio: {}".format(roundleftdiff,roundrightdiff,round(rightdiff/leftdiff,4))
                #cv2.putText(image, textdirection, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            else:
                textdirection =  "direct Right: l: {}, r: {}, ratio: {}".format(roundleftdiff,roundrightdiff,round(rightdiff/leftdiff,4))
                #cv2.putText(image, textdirection, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        else:
            ratio = leftdiff/rightdiff
            if leftdiff/rightdiff >0.5:
                textdirection =  "direct Mid: l: {}, r: {}, ratio: {}".format(roundleftdiff,roundrightdiff,round(leftdiff/rightdiff,4))
                #cv2.putText(image, textdirection, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            else:
                textdirection =  "direct Left, l: {}, r: {}, ratio: {}".format(roundleftdiff,roundrightdiff,round(leftdiff/rightdiff,4))
                #cv2.putText(image, textdirection, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    elif mode == 'difference':
        if leftdiff > rightdiff:
            ratio = leftdiff-rightdiff
            if rightdiff/leftdiff >0.5:
                textdirection =  "direct Mid: l: {}, r: {}, ratio: {}".format(roundleftdiff,roundrightdiff,round(rightdiff/leftdiff,4))
                #cv2.putText(image, textdirection, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            else:
                textdirection =  "direct Right: l: {}, r: {}, ratio: {}".format(roundleftdiff,roundrightdiff,round(rightdiff/leftdiff,4))
                #cv2.putText(image, textdirection, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        else:
            ratio = rightdiff-leftdiff
            if leftdiff/rightdiff >0.5:
                textdirection =  "direct Mid: l: {}, r: {}, ratio: {}".format(roundleftdiff,roundrightdiff,round(leftdiff/rightdiff,4))
                #cv2.putText(image, textdirection, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            else:
                textdirection =  "direct Left, l: {}, r: {}, ratio: {}".format(roundleftdiff,roundrightdiff,round(leftdiff/rightdiff,4))
                #cv2.putText(image, textdirection, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    #elif mode == 'sumratio':
    sum = leftdiff+rightdiff
    leftdiffratio = leftdiff/sum
    if leftdiffratio > standard:
        
        if abs(leftdiffratio-standard) < 0.2:
            textdirection =  "direct Mid: l: {}, r: {}, ratio: {}".format(roundleftdiff,roundrightdiff,round(leftdiffratio,4))
            cv2.putText(image, textdirection, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        else:
            status = "right"
            textdirection =  "direct Right: l: {}, r: {}, ratio: {}".format(roundleftdiff,roundrightdiff,round(leftdiffratio,4))
            cv2.putText(image, textdirection, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    else:
        ratio = leftdiffratio
        if abs(leftdiffratio-standard) < 0.2:
            textdirection =  "direct Mid: l: {}, r: {}, ratio: {}".format(roundleftdiff,roundrightdiff,round(leftdiffratio,4))
            cv2.putText(image, textdirection, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        else:
            status = "left"
            textdirection =  "direct Left, l: {}, r: {}, ratio: {}".format(roundleftdiff,roundrightdiff,round(leftdiffratio,4))
            cv2.putText(image, textdirection, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    return ratio , leftdiffratio, status


def facevertdir(high,mid,bot,image, mode='ratio',standard=0.5):

    upperdiff = high[1] - mid[1]
    roundupper = round(upperdiff,5)
    lowerdiff = mid[1] - bot[1]
    roundlower = round(lowerdiff,5)
    status = "mid"
    if mode == 'ratio':
        if upperdiff > lowerdiff:
            ratio = roundlower/roundupper
            textdirection =  "direct bot: \nl: {}, r: {}, ratio: {}".format(roundupper,roundlower,round(roundlower/roundupper,4))
            #cv2.putText(image, textdirection, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        else:
            ratio = roundupper/roundlower
            textdirection =  "direct top, \nl: {}, r: {}, ratio: {}".format(roundupper,roundlower,round(roundupper/roundlower,4))
            #cv2.putText(image, textdirection, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    elif mode == 'difference':
        if upperdiff > lowerdiff:
            ratio = roundlower-roundupper
            textdirection =  "direct bot: \nl: {}, r: {}, ratio: {}".format(roundupper,roundlower,round(roundlower-roundupper,4))
            #cv2.putText(image, textdirection, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        else:
            ratio = roundupper-roundlower
            textdirection =  "direct top, \nl: {}, r: {}, ratio: {}".format(roundupper,roundlower,round(roundupper-roundlower,4))
            #cv2.putText(image, textdirection, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    #elif mode == 'sumratio':
    sum = upperdiff+lowerdiff
    upperratio = upperdiff/sum

    if upperratio > standard:
        
        """ textdirection =  "direct bot: \nl: {}, r: {}, ratio: {}".format(roundupper,roundlower,round(roundlower/roundupper,4))
        cv2.putText(image, textdirection, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA) """
        if abs(upperratio-standard) < 0.15:
            textdirection =  "direct Mid: l: {}, r: {}, ratio: {}".format(roundupper,roundlower,round(upperratio,4))
            cv2.putText(image, textdirection, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        else:
            status = "bot"
            textdirection =  "direct Bot: l: {}, r: {}, ratio: {}".format(roundupper,roundlower,round(upperratio,4))
            cv2.putText(image, textdirection, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    else:
        
        """ textdirection =  "direct bot: \nl: {}, r: {}, ratio: {}".format(roundupper,roundlower,round(roundlower/roundupper,4))
        cv2.putText(image, textdirection, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA) """
        if abs(upperratio-standard) < 0.15:
            textdirection =  "direct Mid: l: {}, r: {}, ratio: {}".format(roundupper,roundlower,round(upperratio,4))
            cv2.putText(image, textdirection, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        else:
            status = "top"
            textdirection =  "direct Top: l: {}, r: {}, ratio: {}".format(roundupper,roundlower,round(upperratio,4))
            cv2.putText(image, textdirection, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    return ratio , upperratio, status

#0: nose
#left ear : 7
#right_ear : 8
#left wrist: 15
#right wrist : 16
#mouth: 9/ 10
#leye:3
#reye:6

counter = 0
learningPoseSumHori = 0
learningPoseSumVert = 0
counterLearning = 0
standardh = 0.5
standardv = 0.5

QUEUESIZE = 60


#images for frames in 3s
imageQueue = queue.Queue(3*1000//WAITKEY)
imageDataQueue = queue.Queue(3*1000//WAITKEY)

halfminuteImageQueue = queue.Queue(QUEUESIZE)
halfminuteImageDataQueue = queue.Queue(QUEUESIZE)



#https://google.github.io/mediapipe/solutions/pose.html
mp_holistic.POSE_CONNECTIONS

mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    while cap.isOpened():
        ret, frame = cap.read()
        counter += 1
        if counter == 6000:
            counter = 0
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = holistic.process(image)

        image_height, image_width, _ = image.shape

        
        # Recolor image back to BGR for rendering
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        """ print(
          f'Nose coordinates: ('
          f'{results.pose_landmarks.landmark[0].x * image_width}, '
          f'{results.pose_landmarks.landmark[0].y * image_height})'
        ) """

        #0: nose
        #left ear : 7
        #right_ear : 8
        #left wrist: 15
        #right wrist : 16
        #mouth: 9/ 10
        warnFlag = False
        warnMessage = ""
        lefthand = "low"
        righthand = "low"
        #print(results.face_landmarks)

        nose_coor = [results.pose_landmarks.landmark[0].x, results.pose_landmarks.landmark[0].y]
        lear_coor = [results.pose_landmarks.landmark[7].x, results.pose_landmarks.landmark[7].y]
        rear_coor = [results.pose_landmarks.landmark[8].x, results.pose_landmarks.landmark[8].y]
        lwrist_coor = [results.pose_landmarks.landmark[19].x, results.pose_landmarks.landmark[19].y]
        rwrist_coor = [results.pose_landmarks.landmark[20].x, results.pose_landmarks.landmark[20].y]
        mouth_coor = [results.pose_landmarks.landmark[9].x, results.pose_landmarks.landmark[9].y]
        leye_coor = [results.pose_landmarks.landmark[3].x, results.pose_landmarks.landmark[3].y]
        reye_coor = [results.pose_landmarks.landmark[6].x, results.pose_landmarks.landmark[6].y]
        
        leye_upper = [results.face_landmarks.landmark[159].x, results.face_landmarks.landmark[159].y]
        leye_lower = [results.face_landmarks.landmark[145].x, results.face_landmarks.landmark[145].y]
        reye_upper = [results.face_landmarks.landmark[386].x, results.face_landmarks.landmark[386].y]
        reye_lower = [results.face_landmarks.landmark[374].x, results.face_landmarks.landmark[374].y]
        #print('+++++++++++++++++++++++++++++++')
        # 1. Draw face landmarks
        """ mp_drawing.draw_landmarks(
        image,
        results.face_landmarks,
        mp_holistic.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles
        .get_default_face_mesh_contours_style()) """
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                 mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                 )
        #159 - 145
        #386 -374
        """
        # 2. Right hand
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                 )

        # 3. Left Hand
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                 ) """

        # 4. Pose Detections pose_landmarks
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                 )
        
        if lwrist_coor[1] > mouth_coor[1]:
            lefthand = "low"
        
            cv2.putText(image, "lwrist lower", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        else:
            lefthand = "high"
            cv2.putText(image, "lwrist higher", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        
        if rwrist_coor[1] > mouth_coor[1]:
            righthand = "low"
            cv2.putText(image, "rwrist lower", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        else:
            righthand = "high"
            cv2.putText(image, "rwrist higher", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        
        print('leye_upper[1] - leye_lower[1]',leye_upper[1] - leye_lower[1])
        #print('reye_upper[1] - reye_lower[1]',reye_upper[1] - reye_lower[1])
        ##########################################




        ##########################################
        ratioh , leftratio , statusH = facehoridir(leye_coor,reye_coor,nose_coor,image,mode='ratio',standard = standardh)
        if reye_coor[1] > leye_coor[1]:
            lowereye = leye_coor
        else:
            lowereye = reye_coor
        ratiov, upperratio,  statusV = facevertdir(lowereye,nose_coor,mouth_coor,image,mode='ratio',standard = standardv)
        
        textprerecord = "Please ready for your driving pose in {} seconds".format(WAITING_TIME)
        textrecording = "Please keep your driving pose in {} seconds".format(LEARNING_TIME)
        font = cv2.FONT_HERSHEY_SIMPLEX
        textsize = cv2.getTextSize(textprerecord, font, 1, 2)[0]
        #print(image_width)
        #print(textsize[0])
        textX = int((image_width - textsize[0]*0.7) / 2)
        #print(textX)


        if counter //(1000/WAITKEY) < WAITING_TIME:
            textprerecord = "Please ready for your driving pose in {} seconds".format(WAITING_TIME-(counter//(1000/WAITKEY))-1)
            #cv2.putText(image, "rwrist lower", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(image, textprerecord, (textX, 400), font, 0.7, (0, 255, 255), 1, cv2.LINE_AA)
        elif counter // (1000/WAITKEY) < WAITING_TIME+LEARNING_TIME and counter // (1000/WAITKEY) >= WAITING_TIME:
            textrecording = "Please keep your driving pose in {} seconds".format(LEARNING_TIME+WAITING_TIME-(counter//(1000/WAITKEY))-1)
            cv2.putText(image, textrecording, (textX, 400), font, 0.7, (0, 255, 255), 1, cv2.LINE_AA)
            learningPoseSumHori += leftratio
            learningPoseSumVert += upperratio
            counterLearning += 1
        elif counter // (1000/WAITKEY) >= WAITING_TIME+LEARNING_TIME:
            standardh = learningPoseSumHori/counterLearning
            standardv = learningPoseSumVert/counterLearning
            textdetecting = "Concentration is detecting"
            cv2.putText(image, textdetecting, (textX, 400), font, 0.7, (0, 255, 255), 1, cv2.LINE_AA)
        


        #textY = (img.shape[0] + textsize[1]) / 2

        #each seconds
        """ if counter %(1000/WAITKEY) == 0:
            r = requests.get('http://localhost:3001/home')
            response = r.json()
            print(response) """
        #each half seconds
        
        if counter %((1000/WAITKEY)//2) == 0 and counter > 60:
            if imageQueue.full():
                imageQueue.get()
            if imageDataQueue.full():
                imageDataQueue.get()
            if halfminuteImageDataQueue.full():
                halfminuteImageDataQueue.get()   
            imageQueue.put(image)
            imagestatus = [DRIVER_ID,lefthand,righthand,statusH,statusV]
            imageDataQueue.put(imagestatus)
            timestringserver = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            newImageRecord = ImageRecord(DRIVER_ID,image,lefthand,righthand,statusH,statusV,timestringserver, time.time())
            halfminuteImageDataQueue.put(newImageRecord)
            #print(newImageRecord.statusReturn())

            address = 'C:\\Users\\Acer\\Downloads\\CARLA_0.9.12\\WindowsNoEditor\\PythonAPI\\examples\\face_image\\'
            timestring = time.strftime("%Y-%m-%d_%H_%M_%S.jpg", time.localtime())
            filename = address+timestring
            #my_files = {'my_filename': open('./face_image/2022-03-26_01_27_27.jpg', 'rb')}
            #cv2.imwrite(filename, image)

            if counter %(600) == 0:
                r = requests.post('http://localhost:3001/driveraddmins',json={"driver_id":DRIVER_ID})
                response = r.json()
                print(response)
        

            #r = requests.post('http://localhost:3001/faceEvents',json=newImageRecord.statusReturn())
            #cumulate, consec, maxi consec
            lefthandcounter = [0,0,0]
            righthandcounter = [0,0,0]
            leftfacecounter = [0,0,0]
            rightfacecounter = [0,0,0]
            topfacecounter = [0,0,0]
            botfacecounter = [0,0,0]
            for queueloop in range(halfminuteImageDataQueue.qsize()):
                #print('?')
                if halfminuteImageDataQueue.empty():
                    break
                #halfminuteImageDataQueue.queue[queueloop].statusReturn().lefthand
                getrecord =  halfminuteImageDataQueue.queue[queueloop].statusReturn()
                if getrecord['lefthand'] == "high":
                    lefthandcounter[0] += 1
                    lefthandcounter[1] += 1
                    if righthandcounter[1]>righthandcounter[2]:
                        righthandcounter[2] = righthandcounter[1]
                else:
                    if lefthandcounter[1]>lefthandcounter[2]:
                        lefthandcounter[2] = lefthandcounter[1]
                    lefthandcounter[1] = 0
                if getrecord['righthand'] == "high":
                    righthandcounter[0] += 1
                    righthandcounter[1] += 1
                    if righthandcounter[1]>righthandcounter[2]:
                        righthandcounter[2] = righthandcounter[1]
                else:
                    if righthandcounter[1]>righthandcounter[2]:
                        righthandcounter[2] = righthandcounter[1]
                    righthandcounter[1] = 0
                if getrecord['statusH'] == "left":
                    leftfacecounter[0] += 1
                    leftfacecounter[1] += 1
                    if leftfacecounter[1]>leftfacecounter[2]:
                        leftfacecounter[2] = leftfacecounter[1]
                else:
                    if leftfacecounter[1]>leftfacecounter[2]:
                        leftfacecounter[2] = leftfacecounter[1]
                    leftfacecounter[1] = 0
                if getrecord['statusH'] == "right":
                    rightfacecounter[0] += 1
                    rightfacecounter[1] += 1
                    if rightfacecounter[1]>rightfacecounter[2]:
                        rightfacecounter[2] = rightfacecounter[1]
                else:
                    if rightfacecounter[1]>rightfacecounter[2]:
                        rightfacecounter[2] = rightfacecounter[1]
                    rightfacecounter[1] = 0
                if getrecord['statusV'] == "top":
                    topfacecounter[0] += 1
                    topfacecounter[1] += 1
                    if topfacecounter[1]>topfacecounter[2]:
                        topfacecounter[2] = topfacecounter[1]
                else:
                    if topfacecounter[1]>topfacecounter[2]:
                        topfacecounter[2] = topfacecounter[1]
                    topfacecounter[1] = 0
                if getrecord['statusV'] == "bot":
                    botfacecounter[0] += 1
                    botfacecounter[1] += 1
                    if botfacecounter[1]>botfacecounter[2]:
                        botfacecounter[2] = botfacecounter[1]
                else:
                    if botfacecounter[1]>botfacecounter[2]:
                        botfacecounter[2] = botfacecounter[1]
                    botfacecounter[1] = 0


            if getrecord['lefthand'] == "high" and getrecord['righthand'] == "high":
                warnFlag =True
                warnMessage += "Both hands off the steering wheel; "
            if topfacecounter[1] >= 10 or botfacecounter[1] >= 10:
                warnFlag =True
                warnMessage += "Not focusing for 5 seconds(Face Upper/Lower); "
            """ if topfacecounter[0] + botfacecounter[0] >= 30:
                warnFlag =True
                warnMessage += "Not focusing very often (Face Upper/Lower); " """
            """ if lefthandcounter[0] + righthandcounter[0] >= 30:
                warnFlag =True
                warnMessage += "Single hand driving very often (Phone Calling); " """
            if lefthandcounter[1] >= 16 or righthandcounter[1] >= 16:
                warnFlag =True
                warnMessage += "Single hand driving for 8 seconds (Phone Calling); "
            
            if warnFlag: 
                newImageRecord.updateMessage(warnMessage)
                r = requests.post('http://localhost:3001/faceEvents',json=newImageRecord.statusReturn())
                print(warnMessage)
                response = r.json()
                print(response)
                cv2.imwrite(filename, image)
                files =  open(filename, 'rb')
                my_files = {'file':files}
                r2 = requests.post('http://localhost:3001/image/'+response['id'],files=my_files)
                #response2 = r2.json()
                
                #print(response2)


        #imageQueue.put(image)
        #imagestatus = [DRIVER_ID,lefthand,righthand,statusH,statusV]


        #if warnFlag: 
            #for imageloop in imageQueue:
                #address = 'C:\\Users\\Acer\\Downloads\\CARLA_0.9.12\\WindowsNoEditor\\PythonAPI\\examples\\face_image\\'
                #timestring = time.strftime("%Y-%m-%d_%H_%M_%S.jpg", time.localtime())
                #filename = address+timestring
                #r = requests.post('http://localhost:3001/faceEvents',json=newImageRecord.statusReturn())
                #response = r.json()
                #print(response)
                #cv2.imwrite(filename, imageloop)


        cv2.imshow('Raw Webcam Feed', image)
        
        

        #0.5s per frame
        if cv2.waitKey(WAITKEY) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()