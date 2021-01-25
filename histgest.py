import cv2
import numpy as np
import matplotlib.pyplot as plt
import vlc 
import time

vdo = ("test_vdo.mp4", "Faded.mp3")                              ##add ur vdo list here and access via the index
# create video capture object 
data = cv2.VideoCapture(vdo[0]) 
  
# count the number of frames 
frames = data.get(cv2.CAP_PROP_FRAME_COUNT) 
fps = int(data.get(cv2.CAP_PROP_FPS)) 
  
# calculate dusration of the video 
seconds = int(frames / fps) 
milliseconds = seconds*1000
data.release()

a = 500
r = 1.0001
n = 0
direction = "rest"                                                 #direction of rotation. Use this variable with vlc library
media_player = vlc.MediaPlayer() 
media = vlc.Media(vdo[0])
media_player.set_media(media)

first = False

'''All the skin will be detected during the backprojection of our reference image. To avoid our fce being detected in that, lets blacken out the face
by first detecting it using a frontal face haar cascade'''
face_cascade=cv2.CascadeClassifier(r"frontalface.xml")

#Load the template image of the palm
palm1_bgr = cv2.imread("palm1.jpg")
palm1 = cv2.cvtColor(palm1_bgr, cv2.COLOR_BGR2GRAY)                   #we ll use this for template matching also
width, height = palm1.shape[:2]

#Resize the image so that we can view it comfortably in the size of our screen
palm1_bgr = cv2.resize(palm1_bgr, (width//8, height//8))
palm1 = cv2.resize(palm1, (width//8, height//8))

#Since our reference image has white background we use binary inverse method. This may vary if you choose a different image as your template
_, palm1_mask = cv2.threshold(palm1, 220, 255, cv2.THRESH_BINARY_INV)
cnt_ref, hierarchy = cv2.findContours(palm1_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
palm1_jpg = cv2.bitwise_and(palm1_bgr, palm1_bgr, mask = palm1_mask)
template = cv2.cvtColor(palm1_jpg, cv2.COLOR_BGR2GRAY)                       #template
threshold = 0.6

#Capture the histogram of our template for the first two axes. hsv colour space is recommended here since most of the information is capture by the hue and saturation values
palm1_hsv = cv2.cvtColor(palm1_jpg, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(palm1_hsv)
minval_h = np.min(h[np.nonzero(h)])
minval_s = np.min(s[np.nonzero(s)])
minval_v = np.min(v[np.nonzero(v)])
 
palm1_hsv_hist = cv2.calcHist([palm1_hsv], [0, 1], None, [180, 256], [int(minval_h),180, int(minval_s), 256])
cv2.normalize(palm1_hsv_hist, palm1_hsv_hist, 0, 255, cv2.NORM_MINMAX)

cap=cv2.VideoCapture(0)
global frame
while cap.isOpened():
    _, frame = cap.read()
    flip = cv2.flip(frame, 1)
    #detect faces using haar cascades
    faces = face_cascade.detectMultiScale(flip)
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    #Back project the frame using our calculated histogram to identify pixels falling the same category as that of our template image
    mask = cv2.calcBackProject([frame_hsv], [0, 1], palm1_hsv_hist, [1,180, 1, 256], 1)
    kernel = np.ones((5,5), np.float32)
    kernel_ = np.ones((15,15), np.uint8)
    
    #perform filtering and morphological transformations based on your mask output. This output varies depending on your background conditions
    fil = cv2.filter2D(mask, -1, kernel)
    opening = cv2.dilate(fil,kernel_, iterations = 2)
    frame = cv2.bitwise_and(frame, frame, mask = opening)
    #create duplicates for finding and drawing contours
    duplicate = cv2.flip(frame, 1)
    op_duplicate = cv2.flip(opening, 1)
    duplicate_gray = cv2.cvtColor(duplicate, cv2.COLOR_BGR2GRAY)
    minArea = 27000                                                  #This is a factor which can be used to eliminate false detections(non-faces)

    for (x, y, w_, h_) in faces:
        if w_ * h_ >= minArea:          
            cv2.rectangle(duplicate, (x,y), (x + w_, y + h_), (255,255,0), 5)
            #blacken out the image
            op_duplicate[y:y + h_, x:x + w_] = 0
            duplicate_gray[y:y + h_, x:x + w_] = 0
    ret, threshold = cv2.threshold(duplicate_gray, 50, 255, cv2.THRESH_BINARY)
    threshold = cv2.erode(threshold, kernel_, iterations = 2)
    
    #find the contours of the mask image and use area threshold(since hand is closer to webcam) to eliminate false contours(non-hands)
    duplicate_gray = cv2.bitwise_and(duplicate_gray, duplicate_gray, mask = threshold)
    for_contour_bgr = duplicate.copy()
    for_contour = threshold.copy()
    contours, hierarchy = cv2.findContours(for_contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt_area_threshold = 16000
    hull = []

    valid_contours = []
    for cnt in contours:
        if cv2.contourArea(cnt) >= cnt_area_threshold:
            valid_contours.append(cnt)
    #do a further thresholding using cv2.matchShapes
    if len(valid_contours) == 0:
        direction = "rest"
    else:
        ret=[]
        for cnt in valid_contours:
            ret.append(cv2.matchShapes(cnt_ref[-1], cnt, 1, 0.0))
        minpos = ret.index(min(ret))
        hull = []
        indices = []
        hull.append(cv2.convexHull(valid_contours[minpos], returnPoints = False))
        indices.append(cv2.convexHull(valid_contours[minpos], returnPoints = True))
        try:
            defects = cv2.convexityDefects(valid_contours[minpos], hull[-1])
        except:
            pass
        #use fitEllipse to get the angle of rotation
        (x,y), (MA,ma), angle = cv2.fitEllipse(valid_contours[minpos])
        cv2.drawContours(for_contour_bgr, valid_contours, minpos, (0,255,0), 3)
        cv2.drawContours(for_contour_bgr, indices, -1, (255, 0, 0), 3)
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
            start = tuple(valid_contours[minpos][s][0])
            end = tuple(valid_contours[minpos][e][0])
            far = tuple(valid_contours[minpos][f][0])
            cv2.line(for_contour_bgr, start, end, [255, 255, 255], 2)
            cv2.circle(for_contour_bgr, far, 5, [0, 0, 255], -1)
        #Feel free to modify these values based on your background conditions
        if angle >= 160 or angle <= 20:
            direction = "rest"
        elif angle > 20 and angle <= 90:
            direction = "clockwise"
        elif angle < 160 and angle >= 90:
            direction = "anticlockwise"
        
    media_player.play()
    prev = direction
    if direction == "anticlockwise":
        curr_time = media_player.get_time()
        if curr_time < 0:
            media_player.set_time(0)
        else:
            if not first:
                first = True
                n = 0
            n += 1
            media_player.set_time(curr_time - int(a*pow(r, n)))
            media_player.play()
            time.sleep(0.1)
    if direction == "clockwise":
        curr_time = media_player.get_time()
        if curr_time > milliseconds:
            media_player.set_time(0)
        else:
            if not first:
                n = 0
                first = True
            n += 1
            media_player.set_time(curr_time + int(a*pow(r, n)))
            time.sleep(0.1)
            media_player.play()
    if prev != direction:
        first = False
        prev = direction
        n = 0
    cv2.imshow('for_contour_bgr', for_contour_bgr)
    
    k = cv2.waitKey(10) & 0xFF
    if k == 27:
        media_player.stop()
        break
cap.release()
cv2.destroyAllWindows()
