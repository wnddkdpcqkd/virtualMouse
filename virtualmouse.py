import cv2
import numpy as np
import HandTracingModule as htm
import time
import autopy
import random

cap = cv2.VideoCapture(0)

background = cap.read()

#백그라운드 영상 입력
while True:
    (success, background) = cap.read()
    alarm = background.copy()
    cv2.putText(alarm,"press press space for background", (20,30),0,1,(255,0,0),cv2.LINE_4)
    cv2.imshow("background", alarm)
    if cv2.waitKey(33) == 32:
        break

cv2.destroyAllWindows()

#백그라운드 처리
gray_back = cv2.cvtColor(background,cv2.COLOR_BGR2GRAY)

############################
# HSV 색상


def nothing(x):
    print(x);

cv2.namedWindow('trackbar')

cv2.createTrackbar('treshold','trackbar', 0, 255, nothing)
############################


while True:
    (success, img) = cap.read()
    img = cv2.flip(img, 1)

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray_back, gray_img)

    threshold = cv2.getTrackbarPos('treshold','trackbar')
    coon, diff = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    cv2.imshow("backgronud",background)
    cv2.imshow("diff",diff)
    cv2.waitKey(1)




####################################################
def calculateAngle(A, B):

  A_norm = np.linalg.norm(A)
  B_norm = np.linalg.norm(B)
  C = np.dot(A,B)

  angle = np.arccos(C/(A_norm*B_norm))*180/np.pi
  return angle
###########################################################
def distanceBetweenTwoPoints(start, end):
    x1, y1 = start
    x2, y2 = end

    return int(np.sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2)))
###########################################################
def getFingerPosition(max_contour, img_result):
    points1 = []

    # STEP 6-1
    M = cv2.moments(max_contour)

    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    max_contour = cv2.approxPolyDP(max_contour, 0.02 * cv2.arcLength(max_contour, True), True)
    hull = cv2.convexHull(max_contour)

    for point in hull:
        if cy > point[0][1]:
            points1.append(tuple(point[0]))


    cv2.drawContours(img_result, [hull], 0, (0, 255, 0), 2)
    for point in points1:
        cv2.circle(img_result, tuple(point), 15, [0, 0, 0], -1)

    # STEP 6-2

    max_contour2 = np.squeeze(max_contour)
    polygon = Polygon(max_contour2)
    if polygon.is_simple == False:
        return -1, None

    hull = cv2.convexHull(max_contour, returnPoints=False)
    defects = cv2.convexityDefects(max_contour, hull)

    if defects is None:
        return -1, None

    points2 = []
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(max_contour[s][0])
        end = tuple(max_contour[e][0])
        far = tuple(max_contour[f][0])

        angle = calculateAngle(np.array(start) - np.array(far), np.array(end) - np.array(far))

        if angle < 90:
            if start[1] < cy:
                points2.append(start)

            if end[1] < cy:
                points2.append(end)


    cv2.drawContours(img_result, [max_contour], 0, (255, 0, 255), 2)
    for point in points2:
        cv2.circle(img_result, tuple(point), 20, [0, 255, 0], 5)

    # STEP 6-3
    points = points1 + points2
    points = list(set(points))

    # STEP 6-4
    new_points = []
    for p0 in points:

        i = -1
        for index, c0 in enumerate(max_contour):
            c0 = tuple(c0[0])

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

            angle = calculateAngle(np.array(pre) - np.array(p0), np.array(next) - np.array(p0))

            if angle < 90:
                new_points.append(p0)

    return 1, new_points


##############################################################
