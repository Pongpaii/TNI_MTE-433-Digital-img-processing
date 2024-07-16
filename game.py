import cvzone
from cvzone.HandTrackingModule import HandDetector
import cv2
import numpy as np
import os
import random
import time

# Setting camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
hf = 720
wf = 1280
cap.set(3, wf)  # set new width
cap.set(4, hf)  # set new height


def resize2cframe(input_image):  # Resize to camera frame
    hIP, wIP, _ = input_image.shape
    # Resize to camera frame if necessary
    if hIP != hf or wIP != wf:
        input_image = cv2.resize(input_image, (wf, hf))
    return input_image


# Create variable "imgObjects" to keep all object images
# Set initial values to them
pathFolderObject = "GameResources/fruit"
pathListObjectFolder = os.listdir(pathFolderObject)  # folder name of falling objects
# print(pathListObjectFolder)
imgObjects = []
speedObjsX = []
speedObjsY = []
objectPoss = []
points = []

for folder_name in pathListObjectFolder:
    pathSubFolderObject = f"{pathFolderObject}/{folder_name}"
    # print(pathSubFolderObject)
    pathListObject = os.listdir(pathSubFolderObject)  # list of falling objects in subfolder
    # print(pathListObject)
    for path in pathListObject:
        object_filename = os.path.join(pathSubFolderObject, path)
        # print(object_filename)
        imgObject = cv2.imread(object_filename, cv2.IMREAD_UNCHANGED)
        imgObjects.append(imgObject)
        speedObjsX.append(0)
        speedObjsY.append(0)
        objectPoss.append([0, 2000])
        points.append(int(folder_name))  # point = folder name
        imgObjectsFalling = imgObjects.copy()
        imgObjectsFalling_org = imgObjects.copy()
        objectPoss_org = objectPoss.copy()
        ObjectsFallingPoint = points.copy()


def random_object(_num_object):
    global img, score
    for i in range(_num_object):
        if objectPoss[i][1] >= 720 - imgObjectsFalling[i].shape[0]:  # object falls to the bottom
            new_idx = random.randint(0, len(imgObjects) - 1)
            imgObjectsFalling_org[i] = imgObjects[new_idx]
            scaling_size = random.uniform(0.5, 1)
            imgObjectsFalling[i] = cv2.resize(imgObjectsFalling_org[i], (0, 0), fx=scaling_size, fy=scaling_size)
            hObj, wObj, _ = imgObjectsFalling[i].shape
            speedObjsY[i] = random.randint(15, 20)
            speedObjsX[i] = random.choice(
                [0, speedObjsY[i] // 4, speedObjsY[i] // 5, -speedObjsY[i] // 4, -speedObjsY[i] // 5])
            objectPoss[i] = [random.randint(wObj, wf - wObj), 0]  # [x,y]
            ObjectsFallingPoint[i] = points[new_idx]

        # moving object
        hObj, wObj, _ = imgObjectsFalling[i].shape
        objectPoss[i][1] += speedObjsY[i]
        objectPoss[i][1] = np.clip(objectPoss[i][1], 10, 720 - hObj)
        # objectPoss[i][0] += speedObjsX[i]
        # objectPoss[i][0] = np.clip(objectPoss[i][0], 0, 1280 - wObj)
        img = cvzone.overlayPNG(img, imgObjectsFalling[i], objectPoss[i])

        needle_area = blackBg.copy()
        needle_area[needlePos[1]:needlePos[1] + hN, needlePos[0]:needlePos[0] + wN] = imgNeedle[:, :, 3]
        if hands and needle_area.any() > 0:
            object_area = blackBg.copy()
            object_area[objectPoss[i][1]:objectPoss[i][1] + hObj, objectPoss[i][0]:objectPoss[i][0] + wObj] = \
            imgObjectsFalling[i][:, :, 3]
            intersect_area = cv2.bitwise_and(object_area, needle_area, mask=None)
            if intersect_area.any() == 1:
                objectPoss[i][1] = 2000
                score += ObjectsFallingPoint[i]


def reset():
    global objectPoss, score, status, initialTime
    objectPoss = objectPoss_org
    score = 0
    status = "Playing"
    initialTime = time.time()



# Needle image
imgNeedle = cv2.imread("GameResources/needle/cake2.png", cv2.IMREAD_UNCHANGED)

timeLimit = 20

hN = int(0.2 * hf)
imgNeedle = cv2.resize(imgNeedle, ((imgNeedle.shape[1] * hN) // imgNeedle.shape[0], hN))
hN, wN, _ = imgNeedle.shape

# Initial values
img_ratio = 0.2  # transparent level of camera frame (player)
score = 0
needlePos = [0, 0]
status = "Playing"
num_object = 2
blackBg = np.zeros([hf, wf], dtype=np.uint8)

detector = HandDetector(detectionCon=0.8, maxHands=1)
initialTime = time.time()

imgBackground1 = cv2.imread("GameResources/bg/bg33.jpg")
imgBackground2 = cv2.imread("GameResources/bg/1.png")

# Game over image
imgGameover = cv2.imread("GameResources/go.jpg")
imgGameover = resize2cframe(imgGameover)

# Time's up image
imgTimesup = cv2.imread("GameResources/tu.png")
imgTimesup = resize2cframe(imgTimesup)

# Big win image
imgGamewin = cv2.imread("GameResources/win.jpg")
imgGamewin = resize2cframe(imgGamewin)

while True:
    if status == "Game over":
        img = imgGameover.copy()
    elif status == "Time's up":
        if score <= 20:
            img = imgTimesup.copy()
            cv2.putText(img, f"Your score is {score}.", (330, 70), cv2.FONT_HERSHEY_COMPLEX,
                        2, (50, 0, 200), 5, cv2.LINE_AA)  # fontScale, color, line thickness, line type
        else:  # score > 20
            img = imgGamewin.copy()

    elif status == "Playing":

        timer = timeLimit + initialTime - time.time()

        # Get image frame
        success, img = cap.read()
        img = cv2.flip(img, 1)
        hands, img = detector.findHands(img, draw=True, flipType=True)
        if timer < 1:
            status = "Time's up"
        elif score < 0:
            status = "Game over"
        elif score <= 3:
            num_object = 2
            imgBackground = cv2.resize(imgBackground1, (wf, hf))
        elif score > 3:
            num_object = 5
            imgBackground = cv2.resize(imgBackground2, (wf, hf))

        img = cv2.addWeighted(img, img_ratio, imgBackground, 1 - img_ratio, 0)

        random_object(num_object)
        cv2.putText(img, f"TIME LEFT:{int(timer)}", (1050, 40), cv2.FONT_HERSHEY_COMPLEX,
                    1, (255, 255, 255), 5, cv2.LINE_AA)  # fontScale, color, line thickness, line type
        cv2.putText(img, f"TIME LEFT:{int(timer)}", (1050, 40), cv2.FONT_HERSHEY_COMPLEX,
                    1, (255, 50, 50), 2, cv2.LINE_AA)  # fontScale, color, line thickness, line type

        # fontScale, color, line thickness, line type

        if hands:
            hand = hands[0]
            HandLandMarkList = hand["lmList"]
            fingertip_x, fingertip_y = HandLandMarkList[8][0:2]
            needlePos = [fingertip_x - int(wN * 0.5), fingertip_y - int(hN * 0.8)]
            needlePos[0] = np.clip(needlePos[0], 0, wf - wN)
            needlePos[1] = np.clip(needlePos[1], 0, hf - hN)
            img = cvzone.overlayPNG(img, imgNeedle, needlePos)
            img = cv2.circle(img, (fingertip_x, fingertip_y), 10, (255, 0, 255), -1)
            cv2.putText(img, str(score).zfill(2), (20, 70), cv2.FONT_HERSHEY_COMPLEX,
                        2.5, (200, 0, 200), 5, cv2.LINE_AA)

    cv2.imshow("Fruit Game", img)
    key = cv2.waitKey(1)

    if key == 27:  # Esc
        break

    elif key == ord('r'):
        reset()
