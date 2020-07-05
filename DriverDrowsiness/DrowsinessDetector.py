from tkinter import *
import tkinter
from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2


main = tkinter.Tk()
main.title("Driver Drowsiness Monitoring")
main.geometry("500x400")

def EAR(drivereye):
    point1 = dist.euclidean(drivereye[1], drivereye[5])
    point2 = dist.euclidean(drivereye[2], drivereye[4])
    # compute the euclidean distance between the horizontal
    distance = dist.euclidean(drivereye[0], drivereye[3]) #FINDING DISTANCE BETWEEN THE EYES
    # compute the eye aspect ratio
    ear_aspect_ratio = (point1 + point2) / (2.0 * distance) #TOTAL EYE AVERAGE RATIO GIVES ONE POINT THAT IS AVERAGE POINT
    return ear_aspect_ratio

def MOR(drivermouth):
    # compute the euclidean distances between the horizontal
    point   = dist.euclidean(drivermouth[0], drivermouth[6]) 
    # compute the euclidean distances between the vertical
    point1  = dist.euclidean(drivermouth[2], drivermouth[10])
    point2  = dist.euclidean(drivermouth[4], drivermouth[8])
    # taking average
    Ypoint   = (point1+point2)/2.0
    # compute mouth aspect ratio
    mouth_aspect_ratio = Ypoint/point
    return mouth_aspect_ratio
    
def startMonitoring():
    pathlabel.config(text="          Webcam Connected Successfully")
    webcamera = cv2.VideoCapture(0)
    svm_predictor_path = 'SVMclassifier.dat'
    EYE_AR_THRESH = 0.25
    EYE_AR_CONSEC_FRAMES = 10
    MOU_AR_THRESH = 0.75

    COUNTER = 0
    yawnStatus = False
    yawns = 0
    svm_detector = dlib.get_frontal_face_detector() #DETECTS FACES
    svm_predictor = dlib.shape_predictor(svm_predictor_path) #DETECTS EYES AND MOUTH
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"] #TAKES START AND END POINTS OF LEFT EYE
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
    while True:
        ret, frame = webcamera.read() #WHILE LEFT EYE AND RIGHT EYE IS DETECTED THEN WE SHOULD READ
        frame = imutils.resize(frame, width=640) #DESIGNING THE FRAME THAT VIDEO IS DIVIDED IN IMAGES AND FRAMES
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #CONVERTING EACH FRAME INTO GRAY COLOR
        prev_yawn_status = yawnStatus
        rects = svm_detector(gray, 0) #GRAY VALUES ARE FINDING AND STORING AS ARRAYS
        for rect in rects:
            shape = svm_predictor(gray, rect) #PREDICTING FACE OF SHAPE IN GRAY COLOUR
            shape = face_utils.shape_to_np(shape) #CONVERTING GRAY COLOUR INTO NORMAL COLOUR
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            mouth = shape[mStart:mEnd]
            leftEAR = EAR(leftEye) #PASSING VALUES OF LEFTEYE TO EAR FUNCTION
            rightEAR = EAR(rightEye)
            mouEAR = MOR(mouth)
            ear = (leftEAR + rightEAR) / 2.0 #average distances between eyes 
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye) #drawing boundaries to convex hull to right eye or left eye
            mouthHull = cv2.convexHull(mouth)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 255), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 255), 1)
            cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
            if ear < EYE_AR_THRESH: #average distance betwee eyes because different people has different distances between eyes 
                COUNTER += 1  #counter increases morethan 0 means eyes are closing
                cv2.putText(frame, "Eyes Closed ", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) #printing according to axes on window
                if COUNTER >= EYE_AR_CONSEC_FRAMES: #COUNTER VALUE INCREASES TILL 10 MEANS FRAMES FOUND THAT EYES ARE CLOSED
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 50),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                COUNTER = 0  #IF COUNTER==0 VIDEO IS RUNNING
                cv2.putText(frame, "Eyes Open ", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "EAR: {:.2f}".format(ear), (480, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if mouEAR > MOU_AR_THRESH:
                cv2.putText(frame, "Yawning, DROWSINESS ALERT! ", (10, 70),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                yawnStatus = True
                output_text = "Yawn Count: " + str(yawns + 1)
                cv2.putText(frame, output_text, (10,100),cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,0,0),2)
            else:
                yawnStatus = False
            if prev_yawn_status == True and yawnStatus == False:
                yawns+=1
            cv2.putText(frame, "MAR: {:.2f}".format(mouEAR), (480, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame,"Visual Behaviour & Machine Learning Drowsiness Detection @ Drowsiness",(370,470),cv2.FONT_HERSHEY_COMPLEX,0.6,(153,51,102),1)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    cv2.destroyAllWindows()
    webcamera.release()    


  

font = ('times', 16, 'bold')
title = Label(main, text='Driver Drowsiness Monitoring System using Visual\n Behaviour and Machine Learning',anchor=W, justify=LEFT)
title.config(bg='black', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)


font1 = ('times', 14, 'bold')
upload = Button(main, text="Start Behaviour Monitoring Using Webcam", command=startMonitoring)
upload.place(x=50,y=200)
upload.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='DarkOrange1', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=50,y=250)


main.config(bg='chocolate1')
main.mainloop()
