import cv2
import time
import HandTrackingModule as htm

cap = cv2.VideoCapture(0)
prev_time = 0
current_time = 0
detector = htm.HandDetector()
prevList = []


def getHandMovingDirection(prevList, currList):
    prevX = sumX(prevList) / 21
    prevY = sumY(prevList) / 21
    currX = sumX(currList) / 21
    currY = sumY(currList) / 21

    diffX = int(currX - prevX)
    diffY = int(currY - prevY)

    if abs(diffX) > 3:
        if diffX > 0:
            print('right', diffX)
            return ['right', diffX]
        elif diffX < 0:
            print('left', diffX)
            return ['left', diffX]

    if abs(diffY) > 3:
        if diffY > 0:
            print('down', -diffY)
            return ['down', -diffY]
        elif diffY < 0:
            print('up', -diffY)
            return ['up', -diffY]


def sumX(landmarks):
    total = 0

    for i in landmarks:
        total += i[1]

    return total


def sumY(landmarks):
    total = 0

    for i in landmarks:
        total += i[2]

    return total


def handClosed(landmarks):
    tips = []
    pips = []

    for lm in landmarks:
        if lm[0] == 8 or lm[0] == 12 or lm[0] == 16 or lm[0] == 20:
            tips.append(lm)
        if lm[0] == 6 or lm[0] == 10 or lm[0] == 14 or lm[0] == 18:
            pips.append(lm)

    down = 0

    for tip, pip in zip(tips, pips):
        if tip[2] > pip[2]:
            down += 1

    if down > 3:
        return True

    return False


while True:
    success, img = cap.read()  # enable the camera
    img = cv2.flip(img, 1)
    img = cv2.resize(img, (960, 720), interpolation=cv2.INTER_AREA)
    img = detector.findHands(img)
    lmList = detector.find_landmarks(img)

    if prevList:
        getHandMovingDirection(prevList, lmList)

    prevList = lmList

    hand_closed = handClosed(lmList)
    # print(hand_closed)

    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    cv2.imshow('image', img)

    k = cv2.waitKey(1)

    if k == 27:
        cv2.destroyAllWindows()
        break
