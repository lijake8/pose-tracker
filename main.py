## IMPORTS AND SETUP######################################################################
import cv2 as cv
import mediapipe as mp
import numpy as np
from math import floor
import pyttsx3

engine = pyttsx3.init()

mp_drawing = mp.solutions.drawing_utils #get all drawing utils for visualizing poses
mp_pose = mp.solutions.pose #get pose estimation models: https://google.github.io/mediapipe/solutions/pose.html

# #view video feed
# cap = cv.VideoCapture(0) #0 is default cam (webcam) #(0, cv.CAP_DSHOW)
# while cap.isOpened(): #loop thru feed
#     ret, frame = cap.read() #get cur feed from cam. ret is junk var, frame is what we want
#     cv.imshow('mediapipe feed!', frame) #gives popup window on pc

#     if cv.waitKey(10) & 0xFF == ord('q'): #if we hit q key on keyboard, stop the loop and exit feed
#         break

# #stop the vid feed after breaking out of loop
# cap.release()
# cv.destroyAllWindows()


## MAKE DETECTIONS################################################################################

# #view video feed
# cap = cv.VideoCapture(0) #0 is default cam (webcam) #(0, cv.CAP_DSHOW)

# #setup mediapipe instance
# with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose: #access pose estimation model
#     while cap.isOpened(): #loop thru feed
#         ret, frame = cap.read() #get cur feed from cam. ret is junk var, frame is what we want

#         #detect stuff and render
#         #recolor image because opencv auto puts frames in BGR, we need it as RGB to pass into mediapipe
#         image = cv.cvtColor(frame, cv.COLOR_BGR2RGB) #recolor feed
#         image.flags.writeable = False

#         #make detection
#         results = pose.process(image)

#         #recolor back to BGR
#         image.flags.writeable = True
#         image = cv.cvtColor(frame, cv.COLOR_RGB2BGR) 

#         #render detections
#         mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
#                                     mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
#                                     mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
#                                     ) #utilize drawing utility to draw to our image. 4th and 5th prams optional to change visuals

#         cv.imshow('mediapipe feed!', image) #gives popup window on pc

#         if cv.waitKey(10) & 0xFF == ord('q'): #if we hit q key on keyboard, stop the loop and exit feed
#             break

#     #stop the vid feed after breaking out of loop
#     cap.release()
#     cv.destroyAllWindows()


## DETERMINE JOINTS##############################################################################

# #view video feed
# cap = cv.VideoCapture(0) #0 is default cam (webcam) #(0, cv.CAP_DSHOW)

# #setup mediapipe instance
# with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose: #access pose estimation model
#     while cap.isOpened(): #loop thru feed
#         ret, frame = cap.read() #get cur feed from cam. ret is junk var, frame is what we want

#         #detect stuff and render
#         #recolor image because opencv auto puts frames in BGR, we need it as RGB to pass into mediapipe
#         image = cv.cvtColor(frame, cv.COLOR_BGR2RGB) #recolor feed
#         image.flags.writeable = False

#         #make detection
#         results = pose.process(image)

#         #recolor back to BGR
#         image.flags.writeable = True
#         image = cv.cvtColor(frame, cv.COLOR_RGB2BGR) 

#         #extract landmarks
#         try:
#             landmarks = results.pose_landmarks.landmark
#         except:
#             pass

#         #render detections
#         mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
#                                     mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
#                                     mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
#                                     ) #utilize drawing utility to draw to our image. 4th and 5th prams optional to change visuals

#         cv.imshow('mediapipe feed!', image) #gives popup window on pc

#         if cv.waitKey(10) & 0xFF == ord('q'): #if we hit q key on keyboard, stop the loop and exit feed
#             break

#     #stop the vid feed after breaking out of loop
#     cap.release()
#     cv.destroyAllWindows()

# #get array index for specific joints
# left_elbow_index = mp_pose.PoseLandmark.LEFT_ELBOW.value
# left_shoulder_index = mp_pose.PoseLandmark.LEFT_SHOULDER.value
# left_hip_index = mp_pose.PoseLandmark.LEFT_HIP.value

# #get x and y values of the joint
# left_elbow = [landmarks[left_elbow_index].x, landmarks[left_elbow_index].y]
# left_shoulder = [landmarks[left_shoulder_index].x, landmarks[left_shoulder_index].y]
# left_hip = [landmarks[left_hip_index].x, landmarks[left_hip_index].y]

## CALC ANGLES#####################################################################################

def calc_angle(a,b,c): #endpoint, vertex, endpoint (elbow, shoulder, hip) as [x,y] coordinates
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians / np.pi * 180.0)

    #ensure angle is less than 180
    if angle > 180.0:
        angle = 360.0 - angle
    
    return angle

# path = 'vids/test1.mp4' #for testing

#view video feed
cap = cv.VideoCapture(0) #0 is default cam (webcam) #(0, cv.CAP_DSHOW)

counter = 0

#setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose: #access pose estimation model
    while cap.isOpened(): #loop thru feed
        counter += 1
        # print(counter)

        ret, frame = cap.read() #get cur feed from cam. ret is junk var, frame is what we want

        #detect stuff and render
        #recolor image because opencv auto puts frames in BGR, we need it as RGB to pass into mediapipe
        image = cv.cvtColor(frame, cv.COLOR_BGR2RGB) #recolor feed
        image.flags.writeable = False

        #make detection
        results = pose.process(image)

        #recolor back to BGR
        image.flags.writeable = True
        image = cv.cvtColor(frame, cv.COLOR_RGB2BGR) 

        #extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            #get array index for specific joints
            left_elbow_index = mp_pose.PoseLandmark.LEFT_ELBOW.value
            left_shoulder_index = mp_pose.PoseLandmark.LEFT_SHOULDER.value
            left_hip_index = mp_pose.PoseLandmark.LEFT_HIP.value

            #get x and y values of the joint
            left_elbow = [landmarks[left_elbow_index].x, landmarks[left_elbow_index].y]
            left_shoulder = [landmarks[left_shoulder_index].x, landmarks[left_shoulder_index].y]
            left_hip = [landmarks[left_hip_index].x, landmarks[left_hip_index].y]

            #calc angle
            angle = calc_angle(left_elbow, left_shoulder, left_hip)


            if floor(angle) >= 45 and floor(angle) <= 50 and landmarks[left_hip_index].visibility > 0.1: #right position
                engine.say('hold')
                engine.runAndWait()
            elif floor(angle) > 50 and counter % 15 == 0 and landmarks[left_hip_index].visibility > 0.1: #angle too big
                engine.say('lean more')
                engine.runAndWait()
            elif floor(angle) < 45 and counter % 15 == 0 and landmarks[left_hip_index].visibility > 0.1: #angle too small
                engine.say('lean less')
                engine.runAndWait()

            
        except:
            pass

        #render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                    ) #utilize drawing utility to draw to our image. 4th and 5th prams optional to change visuals

        # #mirror the image so it looks more natural
        image = cv.flip(image, 1)
        #display angle on screen
        display_loc = np.multiply(left_shoulder, [640, 480]).astype(int)
        display_loc[0] = 640 - display_loc[0]
        cv.putText(image, 
                    str(int(angle)),
                    tuple(display_loc), 
                    cv.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (255,255,255), 
                    2, 
                    cv.LINE_AA
                    ) #determine where on the screen to display
        
        cv.imshow('mediapipe feed!', image) #gives popup window on pc

        if cv.waitKey(10) & 0xFF == ord('q'): #if we hit q key on keyboard, stop the loop and exit feed
            break

    #stop the vid feed after breaking out of loop
    cap.release()
    cv.destroyAllWindows()
