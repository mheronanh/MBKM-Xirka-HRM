#%% library
import cv2
import numpy as np
import time


#%% Path to face cascade classifier
face_cascade        = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#%% Function
def read_video(path,size):
    #Initialization
    r           = []
    g           = []
    b           = []
    number_frame = 0
    fps_total = []
    real_fps = 0
    found_face      = False
    tracker_init    = False
    
    #Read Video
    video_cap                   =cv2.VideoCapture(path)
    #Start = time.time()
    while (video_cap.isOpened):
    
        ret, img = video_cap.read()
        if not ret:
            break
        grayscale_img   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Try to update face location using tracker
        if found_face and tracker_init :
            found_face, face_box = tracker.update(img)
    # Try to detect new face
        if found_face==False:
            tracker_init = False
            faces        = face_cascade.detectMultiScale(
                        grayscale_img, 
                        scaleFactor     = 1.2,
                        minNeighbors    = 5,
                        minSize         = (30,30)
                        )
            found_face   = len(faces) > 0
    
    # Reset tracker
        if found_face and not tracker_init:
            face_box = faces[0]
            tracker =  cv2.TrackerMOSSE_create()
            tracker.init(img, tuple(face_box))
            tracker_init = True
            
        if found_face:
            #ROI dahi
            x1 = int(face_box[0]+0.2*face_box[2])
            y1 = int(face_box[1] + 0*face_box[3])
            x2 = int(face_box[0] + 0.8*face_box[2])
            y2 = int(face_box[1] + 1*face_box[3])
            
            p1 = (x1,y1)
            p2 = (x2,y2)
            cv2.rectangle(img, p1, p2, (255, 0, 0), 2, 1)
            clipped_roi =img[y1:y2, x1:x2]
            #getting mean value of 3 channel in BGR Format
            color_channel = np.mean(clipped_roi, axis=(0,1))
           
            #Saving value to array
            r.append(np.nan_to_num(color_channel[2]))
            g.append(np.nan_to_num(color_channel[1]))
            b.append(np.nan_to_num(color_channel[0]))
            number_frame += 1
            #print(number_frame)

            #displaying the frames
            image = cv2.resize(img, None, None, fx=0.5, fy=0.4)
            cv2.imshow('video',image)
            if (number_frame==1):
                Start = time.time()
                prev = Start
            else:
                End = time.time()
                fps = 1/(End-prev)
                fps_total.append(fps)
                prev = End
     #Stop the video when ESC key is pressed
        exit = cv2.waitKey(30) & 0xff
        if exit == 27:
            break
    #close all frames
    video_cap.release()
    cv2.destroyAllWindows()
    
    #Calculate real fps
    real_fps = int(number_frame/(End-Start))
    durasi = End-Start
    return r,g,b,durasi,real_fps