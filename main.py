import cv2 as cv
import mediapipe as mp
import numpy as np
from math import floor
import pyttsx3

engine = pyttsx3.init()
mp_drawing = mp.solutions.drawing_utils #get all drawing utils for visualizing poses
mp_pose = mp.solutions.pose #get pose estimation models: https://google.github.io/mediapipe/solutions/pose.html

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

#view video feed
# path = 'vids/test1.mp4' #for testing
video_cap = cv.VideoCapture(0) #0 is default cam (webcam) #(0, cv.CAP_DSHOW) #(path)

counter = 0

#setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose: #access pose estimation model
    while video_cap.isOpened(): #loop thru feed
        counter += 1

        ret, frame = video_cap.read() #get cur feed from cam. ret is junk var, frame is what we want

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

            #get array index for joints
            left_wrist_index = mp_pose.PoseLandmark.LEFT_WRIST.value
            left_elbow_index = mp_pose.PoseLandmark.LEFT_ELBOW.value
            left_shoulder_index = mp_pose.PoseLandmark.LEFT_SHOULDER.value
            left_hip_index = mp_pose.PoseLandmark.LEFT_HIP.value
            left_knee_index = mp_pose.PoseLandmark.LEFT_KNEE.value

            #get x and y values of joints
            left_elbow = [landmarks[left_elbow_index].x, landmarks[left_elbow_index].y]
            left_shoulder = [landmarks[left_shoulder_index].x, landmarks[left_shoulder_index].y]
            left_hip = [landmarks[left_hip_index].x, landmarks[left_hip_index].y]
            left_knee = [landmarks[left_knee_index].x, landmarks[left_knee_index].y]
            left_wrist = [landmarks[left_wrist_index].x, landmarks[left_wrist_index].y]

            #calc angles
            shoulder_angle = calc_angle(left_elbow, left_shoulder, left_hip)
            hip_angle = calc_angle(left_shoulder, left_hip, left_knee)
            elbow_angle = calc_angle(left_wrist, left_elbow, left_shoulder)

            #give voice prompts based on feed
            #shoulder
            if (floor(shoulder_angle) >= 45 and floor(shoulder_angle) <= 50 and landmarks[left_hip_index].visibility > 0.1 
                                                          and landmarks[left_shoulder_index].visibility > 0.1 
                                                          and landmarks[left_elbow_index].visibility > 0.1): #right position
                engine.say('hold')
                engine.runAndWait()
            elif (floor(shoulder_angle) > 50 and counter % 15 == 0 and landmarks[left_hip_index].visibility > 0.1 
                                                          and landmarks[left_shoulder_index].visibility > 0.1 
                                                          and landmarks[left_elbow_index].visibility > 0.1): #shoulder_angle too big
                engine.say('lean more')
                engine.runAndWait()
            elif (floor(shoulder_angle) < 45 and counter % 15 == 0 and landmarks[left_hip_index].visibility > 0.1 
                                                          and landmarks[left_shoulder_index].visibility > 0.1 
                                                          and landmarks[left_elbow_index].visibility > 0.1): #shoulder_angle too small
                engine.say('lean less')
                engine.runAndWait()

            #hip
            if (floor(hip_angle) < 170 and counter % 15 == 5 and landmarks[left_hip_index].visibility > 0.1 
                                                          and landmarks[left_shoulder_index].visibility > 0.1 
                                                          and landmarks[left_knee_index].visibility > 0.1): #piked hip
                engine.say('straighten hips')
                engine.runAndWait()

            #elbow
            if (floor(elbow_angle) < 170 and counter % 15 == 10 and landmarks[left_elbow_index].visibility > 0.1 
                                                          and landmarks[left_shoulder_index].visibility > 0.1 
                                                          and landmarks[left_wrist_index].visibility > 0.1): #piked hip
                engine.say('straighten arms')
                engine.runAndWait()
        except:
            pass

        #render detections on body
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                  ) #utilize drawing utility to draw to our image. 4th and 5th prams optional to change visuals

        #mirror the image so it looks more natural
        image = cv.flip(image, 1)
        
        #display angles on screen
        shoulder_display_location = np.multiply(left_shoulder, [640, 480]).astype(int)
        shoulder_display_location[0] = 640 - shoulder_display_location[0]
        cv.putText(image, 
                   str(int(shoulder_angle)),
                   tuple(shoulder_display_location), 
                   cv.FONT_HERSHEY_SIMPLEX, 
                   1, 
                   (255,255,255), 
                   2, 
                   cv.LINE_AA) 

        hip_display_location = np.multiply(left_hip, [640, 480]).astype(int)
        hip_display_location[0] = 640 - hip_display_location[0]
        cv.putText(image, 
                   str(int(hip_angle)),
                   tuple(hip_display_location), 
                   cv.FONT_HERSHEY_SIMPLEX, 
                   1, 
                   (255,255,255), 
                   2, 
                   cv.LINE_AA) 

        elbow_display_location = np.multiply(left_elbow, [640, 480]).astype(int)
        elbow_display_location[0] = 640 - elbow_display_location[0]
        cv.putText(image, 
                   str(int(elbow_angle)),
                   tuple(elbow_display_location), 
                   cv.FONT_HERSHEY_SIMPLEX, 
                   1, 
                   (255,255,255), 
                   2, 
                   cv.LINE_AA)
        
        cv.imshow('mediapipe feed!', image) #gives popup window on pc

        if cv.waitKey(10) & 0xFF == ord('q'): #if we hit q key on keyboard, stop the loop and exit feed
            break

    #stop the vid feed after breaking out of loop
    video_cap.release()
    cv.destroyAllWindows()
