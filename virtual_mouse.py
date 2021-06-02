import cv2
import numpy as np
import autopy

debug = True
####################################################
def check_inpad(new_points,X,Y,W,H):

    cursor_x,cursor_y = new_points[0]

    if cursor_x > X and cursor_x < X+W and cursor_y > Y and cursor_y < Y+H :
        return True
    else:
        return False

####################################################
def calculateAngle(A, B):
    A_norm = np.linalg.norm(A)
    B_norm = np.linalg.norm(B)
    C = np.dot(A, B)

    angle = np.arccos(C / (A_norm * B_norm)) * 180 / np.pi
    return angle
####################################################
def distanceBetweenTwoPoints(start, end):

    x1,y1 = start
    x2,y2 = end

    return int(np.sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2)))


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

        # if w > h:
        #     continue

        rect_boundary = w*h
        if rect_boundary > max_rect_boundary:
            max_rect_boundary = rect_boundary
            max_area = area
            max_contour = contour
            max_x = x
            max_y = y
            max_w = w
            max_h = h


    if max_area < 100:
        max_area = -1


    return max_area, max_contour,max_x,max_y,max_w,max_h


##############################################################
# cam 크기, 화면 크기
wCam, hCam = 1280, 720
wScr, hScr = autopy.screen.size()
smoothening = 5
plocX, plocY = 0,0
clocX, clocY = 0,0
click_flag = 0
##############################################################

##############################################################

def nothing(x):
    print(x)

def drawTouchPad(x):
    if x == 1 :
        cv2.setTrackbarPos('touchPadX','trackbar1',100)
    elif x == 0 :
        cv2.setTrackbarPos('touchPadX', 'trackbar1', 950)


# HSV 색상 조절
cv2.namedWindow('trackbar')
cv2.createTrackbar('lowH','trackbar', 0, 255, nothing)
cv2.createTrackbar('highH','trackbar', 100, 255, nothing)
cv2.createTrackbar('lowS','trackbar', 29, 255, nothing) # 50
cv2.createTrackbar('highS','trackbar', 255, 255, nothing)
cv2.createTrackbar('lowV','trackbar', 71, 255, nothing) # 70
cv2.createTrackbar('highV','trackbar', 255, 255, nothing)

# trackBar
cv2.namedWindow('trackbar1')
cv2.createTrackbar('setFrame','trackbar1', 640, 1280, nothing)
cv2.createTrackbar('leftRight', 'trackbar1',0,1,drawTouchPad) # 0 : 오른손 , 1 : 왼손
cv2.createTrackbar('touchPadX','trackbar1', 680, 1080, nothing)
cv2.createTrackbar('touchPadY','trackbar1', 100, 960, nothing)
cv2.createTrackbar('touchPadWidth','trackbar1', 480, 540, nothing)
cv2.createTrackbar('touchPadHeight','trackbar1', 320, 480, nothing)



##############################################################


cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

pTime = 0
#detector = htm.handDetector(maxHands=1)

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
        white_mask = np.zeros((720, mask_width), dtype=np.uint8)
        black_mask = np.ones((720, 1280 - mask_width), dtype=np.uint8) * 255
        frame_mask = cv2.hconcat([white_mask, black_mask])





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

    ##########################################################################
    #hsv 이미지 변환 및 threshhold 설정
    hsvImage = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    extract_skin = cv2.inRange(hsvImage, lower, upper)

    ##########################################################################
    #img 에서 skin 부분 추출
    #skin_img : 손 부분만 추출된 gray scale 이미지 , 배경은 흰색
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    reverse_skin = 255 - extract_skin
    skin_img = cv2.add(gray_img,reverse_skin)
    cv2.imshow('skin_img', skin_img)

    ##############################################################
    #frame mask 씌워서 얼굴지움
    hand_mask = cv2.add(skin_img,frame_mask)
    hand_mask = 255 - hand_mask
    # cv2.imshow("mask", hand_mask)

    ##############################################################
    #노이즈 완화하기, blur처리 후 closing 연산
    reduce_noise = cv2.GaussianBlur(hand_mask, (5, 5), 0)
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    for i in range(10):
        reduce_noise = cv2.dilate(reduce_noise, se)
        reduce_noise = cv2.erode(reduce_noise, se)

    cv2.imshow('reduce_noise', reduce_noise)

    ##############################################################
    # 경계값 구분이 애매해서 equalizeHist 로 밝기 분포를 넓힘
    # reduce_noise = cv2.equalizeHist(reduce_noise)
    # cv2.imshow('equalize', reduce_noise)


    ##############################################################
    # Canny Edge로 경계값 검출
    # canny = cv2.Canny(reduce_noise,120,250)
    #cv2.imshow("canny", canny)


    ##############################################################
    # grayscale -> 이진화
    ret, binary = cv2.threshold(reduce_noise, 80, 250, cv2.THRESH_BINARY)
    cv2.imshow("binary", binary)

    ##############################################################
    # contour 검출
    contours, hierachy = cv2.findContours(binary, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)

    ##############################################################
    # 가장 큰 contour 검출, contour box drawing
    max_area, max_contour,x,y,w,h = findMaxArea(contours)

    if max_contour is not None:
        cv2.drawContours(img, [max_contour], 0, (0, 0, 255), 3)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0))

    ##############################################################
    # touchPad 및 인식 범위 설정
    touchPadX = cv2.getTrackbarPos('touchPadX', 'trackbar1')
    touchPadY = cv2.getTrackbarPos('touchPadY', 'trackbar1')
    touchPadWidth = cv2.getTrackbarPos('touchPadWidth', 'trackbar1')
    touchPadHeight = cv2.getTrackbarPos('touchPadHeight', 'trackbar1')

    ##############################################################

    # point 1 -> 추출한 max_contour에서 다각형을 이루는 점을 찍어 convex hull을 통해 다각형을 이루는 포인트들
    points1 = []
    if max_contour is None:
        continue
    M = cv2.moments(max_contour)

    # 이상적으로 손을 추출하면 그 contour의 중앙값이 손바닥이 됨
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    max_contour = cv2.approxPolyDP(max_contour, 0.02 * cv2.arcLength(max_contour, True), True)
    hull = cv2.convexHull(max_contour)

    # convex hull을 통해 추출한 포인트들을 집어넣는다.
    # 손가락은 무조건 손바닥보다 위에 있기 때문에 손바닥 아래에서 구한 convex hull point들은 버린다.
    for point in hull:
        if cy > point[0][1]:
            points1.append(tuple(point[0]))
            # 검은색으로 표시되며, convex hull을 이루는 점이다.
    if debug:
        cv2.drawContours(img, [hull], 0, (0, 255, 0), 2)
        for point in points1:
            cv2.circle(img, tuple(point), 15, [0, 0, 0], -1)

    # point 2 -> 손가락의 위치에 해당되는 포인트를 구한다.
    hull = cv2.convexHull(max_contour, returnPoints=False)
    hull[::-1].sort(axis=0)

    # 손의 contour와 convex hull을 비교하여 손의 contour의 오목한 값, 즉 손마디를 defects를 통해 찾아낸다.
    defects = cv2.convexityDefects(max_contour, hull)

    points2 = []
    defects_cnt = 0
    if defects is not None:
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(max_contour[s][0])
            end = tuple(max_contour[e][0])
            far = tuple(max_contour[f][0])

            # defect와 주변의 convex hull을 이루는 point들의 사잇갓을 구한다.
            angle = calculateAngle(np.array(start) - np.array(far), np.array(end) - np.array(far))

            # defect 주변의 양 두 점과의 사잇각이 90도 이하이면 해당 값이 손가락이 위치가 된다.
            if angle < 60:
                defects_cnt += 1
                cv2.circle(img,far,5,(0,255,0),-1)
                if start[1] < cy:
                    points2.append(start)

                if end[1] < cy:
                    points2.append(end)
        # 손가락은 초록색 원으로 그려진다.
        points2 = list(set(points2))
        points2.sort(key=lambda x: x[0])

        if debug:
            cv2.drawContours(img, [max_contour], 0, (255, 0, 255), 2)

            i = 1
            for point in points2:
                cv2.putText(img, str(i), tuple((point[0], point[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
                cv2.circle(img, tuple(point), 20, [0, 255, 0], 5)
                i = i + 1

    points = points1 + points2
    points = list(set(points))

    new_points = []
    # 이전에 구한 모든 포인트들을 iteration하여 가장 높은 위치에 있는 손끝 위치를 구한다.
    for p0 in points:
        i = -1
        for index, c0 in enumerate(max_contour):
            c0 = tuple(c0[0])

            # contour를 이루는 점과 손가락 point와 가장 가까운 값을 찾음
            if p0 == c0 or distanceBetweenTwoPoints(p0, c0) < 20:
                i = index
            break

        if i >= 0:
            pre = i - 1
            if pre < 0:
                pre = max_contour[len(max_contour) - 1][0]
            else:
                pre = max_contour[i - 1][0]

            next = i + 1
            if next > len(max_contour) - 1:
                next = max_contour[0][0]
            else:
                next = max_contour[i + 1][0]

            if isinstance(pre, np.ndarray):
                pre = tuple(pre.tolist())
            if isinstance(next, np.ndarray):
                next = tuple(next.tolist())

            # 그 위치를 기준으로 사이 값을 구하여 90도 이하인 값을 추가함.
            # 이상적으로 가장 높은 위치에 있는 포인트만 양 옆 포인트와의 사잇값이 90도 이하일 것이므로 가장 높은 손끝을 찾는 행위가 된다.
            angle = calculateAngle(np.array(pre) - np.array(p0), np.array(next) - np.array(p0))

            if angle < 90:
                new_points.append(p0)



    #############################################################
    # 가장 높은 위치에 있는 손끝은 보라색으로 표시된다.
    if ret > 0 and len(new_points) > 0:
        for point in new_points:
            cv2.circle(img, point, 20, [255, 0, 255], 5)

    ##############################################################






    ########################################################################
    # 오른손인식 왼손인식 box 그리기
    if left_right == 1 :
        cv2.rectangle(img,(0,0),(mask_width,720),(255,0,0),3)
    elif left_right == 0 :
        cv2.rectangle(img, (mask_width, 0), (1280, 720), (255, 0, 0),3)

    # touchpad 부분 box 그리기
    cv2.rectangle(img, (touchPadX, touchPadY), (touchPadX + touchPadWidth, touchPadY + touchPadHeight), (0, 0, 255),3)

    ########################################################################

    finger_cnt = len(points2)
    if len(new_points) != 0:
        finger_x, finger_y = new_points[0]

        if check_inpad(new_points, touchPadX, touchPadY, touchPadWidth, touchPadHeight) :

            if finger_cnt == 2 and defects_cnt == 1:
                #autopy.mouse.click()
                click_flag += 1
                print(click_flag)
            if finger_cnt == 0 and defects_cnt == 0 :
                cursor_x = np.interp(finger_x, ( touchPadX, touchPadX + touchPadWidth ), (0,wScr))
                cursor_y = np.interp(finger_y, ( touchPadY, touchPadY + touchPadHeight ), (0, hScr))

                clocX = plocX + (cursor_x - plocX) / smoothening
                clocY = plocY + (cursor_y - plocY) / smoothening

                autopy.mouse.move(clocX,clocY)
                if click_flag != 0 :
                    autopy.mouse.click()
                    click_flag = 0

                plocX, plocY = clocX, clocY


    cv2.imshow("contour", img)

    cv2.waitKey(1)





