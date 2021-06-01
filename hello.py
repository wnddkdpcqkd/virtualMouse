import cv2
import numpy as np
import HandTracingModule as htm
import time
import autopy


####################################################
def findMaxArea(contours):
    max_contour = None
    max_area = -1
    max_rect_boundary = -1
    max_x = -1
    max_y = -1
    max_w = -1
    max_h = -1

    for contour in contours:
        area = cv2.contourArea(contour)

        x, y, w, h = cv2.boundingRect(contour)

        if w > h:
            continue

        rect_boundary = w*h
        if rect_boundary > max_rect_boundary:
            max_rect_boundary = rect_boundary
            max_area = area
            max_contour = contour
            max_x = x
            max_y = y
            max_w = w
            max_h = h
        # if (w * h) * 0.4 > area:
        #     continue
        #
        # if w > h:
        #     continue

        # if area > max_area:
        #     max_area = area
        #     max_contour = contour

    if max_area < 100:
        max_area = -1


    return max_area, max_contour,max_x,max_y,max_w,max_h
####################################################


############################
# cam 크기, 화면 크기
wCam, hCam = 1280, 720
wScr, hScr = autopy.screen.size()
############################

##############################################################

def nothing(x):
    print(x)

# HSV 색상 조절
cv2.namedWindow('trackbar')
cv2.createTrackbar('lowH','trackbar', 0, 255, nothing)
cv2.createTrackbar('highH','trackbar', 218, 255, nothing)
cv2.createTrackbar('lowS','trackbar', 0, 255, nothing)
cv2.createTrackbar('highS','trackbar', 106, 255, nothing)
cv2.createTrackbar('lowV','trackbar', 63, 255, nothing)
cv2.createTrackbar('highV','trackbar', 132, 255, nothing)

#
cv2.namedWindow('trackbar1')
cv2.createTrackbar('setFrame','trackbar1', 540, 1280, nothing)
cv2.createTrackbar('leftRight', 'trackbar1',0,1,nothing) # 0 : 오른손 , 1 : 왼손
cv2.createTrackbar('touchPadX','trackbar1', 950, 1080, nothing)
cv2.createTrackbar('touchPadY','trackbar1', 180, 960, nothing)
cv2.createTrackbar('touchPadWidth','trackbar1', 320, 540, nothing)
cv2.createTrackbar('touchPadHeight','trackbar1', 180, 480, nothing)

##############################################################


cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

pTime = 0
detector = htm.handDetector(maxHands=1)

while True:

    #빈 화면 만들기
    mask_width = cv2.getTrackbarPos('setFrame','trackbar1')
    left_right = cv2.getTrackbarPos('leftRight', 'trackbar1')

    #오른손잡이, 왼손잡이 설정
    if left_right == 0:
        white_mask = np.ones((720, mask_width), dtype=np.uint8) * 255
        black_mask = np.zeros((720, 1280 - mask_width), dtype=np.uint8)
        frame_mask = cv2.hconcat([white_mask, black_mask])
    elif left_right == 1:
        white_mask = np.zeros((720, mask_width), dtype=np.uint8) * 255
        black_mask = np.ones((720, 1280 - mask_width), dtype=np.uint8)
        frame_mask = cv2.hconcat([white_mask, black_mask])


    cv2.imshow('frame_mask',frame_mask)


    #1. 손가락 특징 찾기



    #11. FrameRate
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    ##########################################################################
    #trackbar
    lowH = cv2.getTrackbarPos('lowH','trackbar')
    highH = cv2.getTrackbarPos('highH', 'trackbar')
    lowS = cv2.getTrackbarPos('lowS', 'trackbar')
    highS = cv2.getTrackbarPos('highS', 'trackbar')
    lowV = cv2.getTrackbarPos('lowV', 'trackbar')
    highV = cv2.getTrackbarPos('highV', 'trackbar')

    lower = np.array([lowH, lowS, lowV], dtype="uint8")
    upper = np.array([highH, highS, highV], dtype="uint8")
    ##########################################################################


    (success, img) = cap.read()
    img = cv2.flip(img,1)

    #hsv 이미지 변환 및 threshhold 설정
    hsvImage = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    extract_skin = cv2.inRange(hsvImage, lower, upper)

    #img 에서 skin 부분 추출
    #skin_img : 손 부분만 추출된 gray scale 이미지 , 배경은 흰색
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    reverse_skin = 255 - extract_skin
    skin_img = cv2.add(gray_img,reverse_skin)
    cv2.imshow('skin_img', skin_img)
    ##############################################################
    #배경색이 흰색이라 검은색으로 바꿔줌
    skin_img = 255 - skin_img
    hand_mask = skin_img + frame_mask
    cv2.imshow("mask", hand_mask)
    ##############################################################
    #노이즈 완화하기, blur처리 후 closing 연산
    reduce_noise = cv2.GaussianBlur(skin_img, (5, 5), 0)
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    for i in range(10):
        reduce_noise = cv2.dilate(reduce_noise, se)
        reduce_noise = cv2.erode(reduce_noise, se)

    #cv2.imshow('reduce_noise', reduce_noise)

    ##############################################################
    # 경계값 구분이 애매해서 equalizeHist 로 밝기 분포를 넓힘
    # reduce_noise = cv2.equalizeHist(reduce_noise)
    # cv2.imshow('equalize', reduce_noise)


    ##############################################################
    # Canny Edge로 경계값 검출
    canny = cv2.Canny(reduce_noise,120,250)
    #cv2.imshow("canny", canny)


    ##############################################################
    # grayscale -> 이진화
    ret, binary = cv2.threshold(canny, 125, 250, cv2.THRESH_BINARY)
    #cv2.imshow("binary", binary)


    ##############################################################
    # contour 검출

    contours, hierachy = cv2.findContours(binary, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_NONE)

    ##############################################################
    # 가장 큰 contour 검출
    max_area, max_contour,x,y,w,h = findMaxArea(contours)

    cv2.drawContours(img, [max_contour], 0, (0, 0, 255), 3)




    ##############################################################
    # touchPad 및 인식 범위 설정
    touchPadX = cv2.getTrackbarPos('touchPadX', 'trackbar1')
    touchPadY = cv2.getTrackbarPos('touchPadY', 'trackbar1')
    touchPadWidth = cv2.getTrackbarPos('touchPadWidth', 'trackbar1')
    touchPadHeight = cv2.getTrackbarPos('touchPadHeight', 'trackbar1')

    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0))
    cv2.rectangle(img, (touchPadX, touchPadY), (touchPadX+touchPadWidth, touchPadY+touchPadHeight), (0, 0, 255),3)

    #cv2.imshow("contour", img)



    # for i in range(len(contours)):
    #     cv2.drawContours(img, contours,i,(0,0,255),3)
    # cv2.imshow("contour" , img)


    cv2.waitKey(1)






