#Libraries
from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np
import time
import dlib
import cv2


def alarm(msg):
    global alarm_status
    global alarm_status2
    global saying
    while alarm_status:
        if alarm_status2:
            saying = True
            saying = False
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear
#defining eyes locations
def final_ear(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)
    ear = (leftEAR + rightEAR) / 2.0
    return (ear, leftEye, rightEye)

def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))
    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))
    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)
    distance = abs(top_mean[1] - low_mean[1])
    return distance
def car_level(level, point):

    cv2.putText(frame, "SYI: {}".format(point), (370, 400),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    if (level == 1):
        cv2.putText(frame, "GUVENLI", (370, 430),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    if (level == 2):
        cv2.putText(frame, "DUSUK RISKLI", (370, 430),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    if (level == 3):
        cv2.putText(frame, "YUKSEK RISKLI", (370, 430),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

foo_counter = 0
EYE_AR=0
EYE_AR_THRESH = 0
EYE_AR_CONSEC_FRAMES_HDROWSY = 25 #Kapalı kalan Frame Sayısı
EYE_AR_CONSEC_FRAMES_LDROWSY = 20
EYE_AR_CONSEC_FRAMES_BLINK = 3 #Kapalı kalan Frame Sayısı
YAWN_THRESH = 0
alarm_status = False
alarm_status2 = False
saying = False
COUNTER = 0
TOTAL_Blink = 0
Drowsy_Blink = 0
SYI = 100
level = 1
perclos = float(0)
closed_eye_frame = 0

print("-> Loading the predictor and detector...")
detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

print("-> Starting Video Stream")
vs = cv2.VideoCapture(0)

frame_count=0
result = cv2.VideoWriter('output.AVI',
                             cv2.VideoWriter_fourcc(*'MJPG'), 30
                             , (600, 450))
print("-> Calculation eye aspect ratio keep your eyes open:")
start_time = time.time()
while True:
    frame_count = frame_count+1
    foo_counter = foo_counter+1
    end_time = time.time()
    duration = int(end_time - start_time)
    start = time.time()
    ret,frame = vs.read()
    frame = cv2.resize(frame,(600,450))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        eye = final_ear(shape)
        ear = eye[0]
        leftEye = eye[1]
        rightEye = eye[2]

        distance = lip_distance(shape)
        if foo_counter <2:
            EYE_AR_THRESH = ear*70/100
            YAWN_THRESH = 33.0


        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        lip = shape[48:60]
        cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)

        if ear < EYE_AR_THRESH:
            COUNTER += 1


            if COUNTER >= EYE_AR_CONSEC_FRAMES_HDROWSY:
                Drowsy_Blink =+ 1
                closed_eye_frame = closed_eye_frame +25
                if alarm_status == False:
                    alarm_status = True
                    SYI = SYI -25
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


        else:
            if COUNTER>=EYE_AR_CONSEC_FRAMES_BLINK:
                TOTAL_Blink += 1
                closed_eye_frame = closed_eye_frame +3
                COUNTER = 0
                alarm_status = False

        if (distance > YAWN_THRESH):

            cv2.putText(frame, "Yawn Alert", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if alarm_status2 == False and saying == False:
                alarm_status2 = True
                SYI = SYI -10
        else:
            alarm_status2 = False
        #Level Belirleme Bloğu
        if SYI > 75:
            level=1
        if SYI <=75  & SYI >=50 :
            level =2
        if SYI <=25:
            level =3
        if (SYI <=0):
            SYI = 0
        car_level(level,SYI)

        if duration %10 == 0:
            if TOTAL_Blink != 0:
                perclos = ((closed_eye_frame)/frame_count)*10
            else:
                perclos = float(0)
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (450, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "YAWN: {:.2f}".format(distance), (450, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Time: {}".format(duration),(10,420),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
        cv2.putText(frame, "Blinks: {}".format(TOTAL_Blink),(450,90),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
        frame_count = frame_count + 1
        cv2.putText(frame, "PERCLOS: {:.2f}".format(perclos), (10, 440),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    result.write(frame)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
    end = time.time()
    time1 = end-start
    frame_count = frame_count+1

    cv2.putText(frame, "Time: {:.2f}".format(time1),(250,250),
            cv2.FONT_HERSHEY_COMPLEX, 0.7,(0,0,255),2)
vs.release()
result.release()
print('Video was Saved')

cv2.destroyAllWindows()
