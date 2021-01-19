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
#np.set_printoptions(threshold=sys.maxsize)


face_cascade=cv2.CascadeClassifier(r"frontalface.xml")


palm1_bgr = cv2.imread("palm1.jpg")
palm1=cv2.cvtColor(palm1_bgr, cv2.COLOR_BGR2GRAY)                   #we ll use this for template matching also
width, height = palm1.shape[:2]
palm1_bgr=cv2.resize(palm1_bgr, (width//8, height//8))
palm1=cv2.resize(palm1, (width//8, height//8))

_, palm1_mask = cv2.threshold(palm1, 220, 255, cv2.THRESH_BINARY_INV)

cnt_ref, hierarchy = cv2.findContours(palm1_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
palm1_jpg = cv2.bitwise_and(palm1_bgr, palm1_bgr, mask=palm1_mask)
template=cv2.cvtColor(palm1_jpg, cv2.COLOR_BGR2GRAY)
#template=palm1_mask
threshold=0.6


palm1_hsv = cv2.cvtColor(palm1_jpg, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(palm1_hsv)
minval_h = np.min(h[np.nonzero(h)])
minval_s = np.min(s[np.nonzero(s)])
minval_v = np.min(v[np.nonzero(v)])
 
palm1_hsv_hist=cv2.calcHist([palm1_hsv], [0, 1], None, [180, 256], [int(minval_h),180, int(minval_s), 256])


cv2.normalize(palm1_hsv_hist, palm1_hsv_hist, 0, 255, cv2.NORM_MINMAX)
#plt.imshow(palm1_hsv_hist)
#plt.show()
#cap=cv2.VideoCapture(0)
#while cap.isOpened():
#    ret, frame = cap.read()

#capture background and eliminate while registering hand

cap=cv2.VideoCapture(0)
global frame, count
count = 0
while cap.isOpened():
    _, frame = cap.read()
    #frame=cv2.resize(frame, (frame.shape[0]*2, frame.shape[1]*2))
    flip = cv2.flip(frame, 1)
    faces=face_cascade.detectMultiScale(flip)
    frame_hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask=cv2.calcBackProject([frame_hsv], [0, 1], palm1_hsv_hist, [1,180, 1, 256], 1)
    kernel=np.ones((5,5), np.float32)
    kernel_=np.ones((15,15), np.uint8)
    
    fil=cv2.filter2D(mask, -1, kernel)
    #gaus=cv2.GaussianBlur(mask, (5,5), 0)
    #med=cv2.medianBlur(mask, 5)
    #bil=cv2.bilateralFilter(mask, 15, 75, 75)
    
    #opening=cv2.erode(fil, kernel_, iterations=1)
    opening = cv2.dilate(fil,kernel_,iterations = 2)
    #opening = cv2.morphologyEx(fil, cv2.MORPH_OPEN, kernel_, iterations=2)
    #opening = cv2.morphologyEx(fil, cv2.MORPH_CLOSE, kernel_, iterations=2)
    #opening=cv2.adaptiveThreshold(opening, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
    #_, opening=cv2.threshold(opening, 12, 255, cv2.THRESH_BINARY)
    #frame=cv2.bitwise_and(frame, frame, mask=dilation)
    frame=cv2.bitwise_and(frame, frame, mask=opening)
    #frame=cv2.bitwise_and(frame, frame, mask=dilation)
    duplicate=cv2.flip(frame, 1)
    op_duplicate=cv2.flip(opening, 1)
    duplicate_gray = cv2.cvtColor(duplicate, cv2.COLOR_BGR2GRAY)
    #faces=face_cascade.detectMultiScale(duplicate)
    minArea=27000
    #print(op_duplicate.shape)
    #print(duplicate_gray.shape)
    for (x,y,w_,h_) in faces:
        if w_*h_>=minArea:
            cv2.rectangle(duplicate, (x,y), (x+w_, y+h_), (255,255,0), 5)
            op_duplicate[y:y+h_, x:x+w_]=0
            duplicate_gray[y:y+h_, x:x+w_]=0
    ret, threshold = cv2.threshold(duplicate_gray, 50, 255, cv2.THRESH_BINARY)
    threshold=cv2.erode(threshold, kernel_, iterations=2)
    duplicate_gray=cv2.bitwise_and(duplicate_gray, duplicate_gray, mask=threshold)
    #res=cv2.matchTemplate(duplicate_gray, template, cv2.TM_CCOEFF_NORMED)
    #print(np.amax(res))
    #loc=np.where(res>=threshold)
    
    #for pt in zip(*loc[::-1]):
    #    cv2.rectangle(duplicate_gray, pt, (min(pt[0]+width//4, duplicate_gray.shape[0]), min(pt[1]+height//4, duplicate_gray.shape[1])), (0,255,0), 2)



    #cv2.imshow('erosion', erosion)
    #cv2.imshow('dilation', dilation)
    for_contour_bgr=duplicate.copy()
    for_contour = threshold.copy()
    w, h = for_contour.shape[:2]
    
    for_contour[:, :(h*9)//16] = 0
    for_contour[:w//8, h//2:] = 0
    for_contour[(w*7)//8 :, h//2:] = 0

    for_contour_bgr[:, :(h*9)//16, :] = 0
    for_contour_bgr[:w//8, h//2:, :] = 0
    for_contour_bgr[(w*7)//8 :, h//2:, :] = 0
    
    contours, hierarchy = cv2.findContours(for_contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt_area_threshold=16000
    hull=[]
    #print(len(contours))
    valid_contours=[]
    for cnt in contours:
        if cv2.contourArea(cnt)>=cnt_area_threshold:
            valid_contours.append(cnt)
    if len(valid_contours)==0:
        direction="rest"
    else:
        ret=[]
        for cnt in valid_contours:
            ret.append(cv2.matchShapes(cnt_ref[-1], cnt, 1, 0.0))
        minpos=ret.index(min(ret))

        #print([cnt])
        #M = cv2.moments(valid_contours[minpos])
        #cx = int(M['m10']/M['m00'])
        #cy = int(M['m01']/M['m00'])
        #leftmost = tuple(contours[j][contours[j][:,:,0].argmin()][0])
        #rightmost = tuple(contours[j][contours[j][:,:,0].argmax()][0])
        #topmost = tuple(contours[j][contours[j][:,:,1].argmin()][0])
        #bottommost = tuple(contours[j][contours[j][:,:,1].argmax()][0])
        #(x,y),(MA,ma),angle = cv2.fitEllipse(valid_contours[minpos])
        hull=[]
        indices=[]
        hull.append(cv2.convexHull(valid_contours[minpos], returnPoints=False))
        indices.append(cv2.convexHull(valid_contours[minpos], returnPoints=True))
        try:
            defects = cv2.convexityDefects(valid_contours[minpos],hull[-1])
        except:
            pass
        (x,y),(MA,ma),angle = cv2.fitEllipse(valid_contours[minpos])
        #print(contours[j], hull)
        cv2.drawContours(for_contour_bgr, valid_contours, minpos, (0,255,0), 3)
        cv2.drawContours(for_contour_bgr, indices, -1, (255, 0, 0), 3)
        #cv2.circle(for_contour_bgr, (cx,cy), 5, (0,0,0), -1)
        #for_contour_bgr=cv2.ellipse(for_contour_bgr, (int(x),int(y)), (int(MA), int(ma)), angle, 0, 360, (255,255,0), 2)
        #print(angle)
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
            start = tuple(valid_contours[minpos][s][0])
            end = tuple(valid_contours[minpos][e][0])
            far = tuple(valid_contours[minpos][f][0])
            cv2.line(for_contour_bgr,start,end,[255,255,255],2)
            cv2.circle(for_contour_bgr,far,5,[0,0,255],-1)
        if angle>=160 or angle<=20:
            direction="rest"
        elif angle>20 and angle<=90:
            direction="clockwise"
        elif angle<160 and angle>=90:
            direction="anticlockwise"
        
    media_player.play()
    prev = direction
    if direction == "anticlockwise":
        curr_time = media_player.get_time()
        if curr_time < 0:
            media_player.set_time(0)
        else:
            if not first:
                first = True
                n=0
                #r = 1 - a/curr_time
            n+=1
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
                #r = 1 - curr_time/milliseconds if curr_time!=0 else 1 - a/milliseconds
            n+=1
            media_player.set_time(curr_time + int(a*pow(r, n)))
            time.sleep(0.1)
            media_player.play()
    if prev != direction:
        first = False
        prev = direction
        n = 0

    cv2.imshow('for_contour_bgr', for_contour_bgr)
    #cv2.imshow('closing', closing)
    
    k=cv2.waitKey(10) & 0xFF
    if k==27:
        media_player.stop()
        break
cap.release()
cv2.destroyAllWindows()
