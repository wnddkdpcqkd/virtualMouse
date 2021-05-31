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