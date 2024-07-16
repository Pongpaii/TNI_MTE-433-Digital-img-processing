import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
from cvzone.FaceDetectionModule import FaceDetector
import numpy as np
import random
import os

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # change index to your hardware camera

detectorF = FaceDetector(minDetectionCon=0.5, modelSelection=0)
detectorH = HandDetector(detectionCon=0.8, maxHands=10)

#faceMask = cv2.imread("Photo2/FaceHandMask/Face/ironman.png", cv2.IMREAD_UNCHANGED)
#handMask = cv2.imread("Photo2/FaceHandMask/Hand/tasmil.png", cv2.IMREAD_UNCHANGED)
#handMask_org = handMask.copy()

pathFolderFaceMask = "resource/FacehandMask/Face"
pathListFaceMask = os.listdir(pathFolderFaceMask)  # path of FaceMask
faceMasks = []
for path in pathListFaceMask:
    faceMask_filename = os.path.join(pathFolderFaceMask, path)
    faceMask = cv2.imread(faceMask_filename, cv2.IMREAD_UNCHANGED) # 3 layers or png
    faceMasks.append(faceMask)

pathFolderHandMask = "resource/FacehandMask/Hand"
pathListHandMask = os.listdir(pathFolderHandMask) # path of HandMask
handMasks = []
for path in pathListHandMask:
    handMask_filename = os.path.join(pathFolderHandMask, path)
    handMask = cv2.imread(handMask_filename, cv2.IMREAD_UNCHANGED) # 3 layers or
    handMasks.append(handMask)

f_th = random.randint(0, len(faceMasks) - 1) # nth face mask image
h_th = random.randint(0, len(handMasks) - 1) # nth hand mask image
def mask_filter(cartoon,fh): # fh: face or hand
    global img, bbox
    x, y, w, h = bbox
    if fh == "face":
        y = int(y - (0.2 * h))
        x = int(x - (0.1 * w))
        w = int(1.2 * w)
        h = int(1.2 * h)
    if x < 0: x = 0
    if y < 0: y = 0
    crop = img[y:y + h, x:x + w]
    hFH, wFH, _ = crop.shape
    hC, wC, cC = cartoon.shape
    cartoon = cv2.resize(cartoon, (wFH, int(wFH*(hC/wC))))
    hC, wC, cC = cartoon.shape
    if fh == "face":
        y1 = y
    elif fh == "hand":
        y1 = int(y+((h-hC)//2))
    x  = np.clip(x, 0, img.shape[1] - wC)
    y1 = np.clip(y1, 0, img.shape[0] - hC)
    if cC == 3: # 3 layers
        img[y1:y1 + hC, x:x + wC] = cartoon
    elif cC == 4: # 4 layers
        img = cvzone.overlayPNG(img, cartoon, (x, y1))

while True:
    _, img = cap.read()
    img = cv2.flip(img, 1)

    img, bboxs = detectorF.findFaces(img, draw=False)
    hands, img = detectorH.findHands(img, draw=False, flipType=True)
    if bboxs:
        for face in bboxs:
            bbox = face['bbox']
            #x, y, w, h = bbox
            #cartoon = cv2.resize(faceMask, (w, h))
            #img = cvzone.overlayPNG(img, cartoon, (x, y))
            cartoonF = faceMasks[f_th]
            mask_filter(cartoonF, "face")

    if hands:
        for hand in hands:
            finger = detectorH.fingersUp(hand)  # [thumb, index finger, middle finger, ring finger, pinky]
            finger = detectorH.fingersUp(hand)  # [thumb, index finger, middle finger, ring finger, pinky]
            # finger == [0, 1, 1, 0, 0] and hand['type'] == 'Left':  # victory pose of right hand
            #    handMask = faceMask
            #else:
            #    handMask = handMask_org
            if finger == [0, 1, 1, 0, 0] and hand['type'] == 'Left':  # victory pose of right hand
                f_th = (f_th + 1) % len(faceMasks)  # loop face mask
            elif finger == [0, 1, 1, 0, 0] and hand['type'] == 'Right':  # victory pose of left hand
                h_th = (h_th + 1) % len(handMasks)  # loop hand mask
            elif finger == [1, 1, 0, 0, 1] and hand['type'] == 'Right':  # victory pose of left hand
                f_th = 2  # loop hand mask
            bbox = hand['bbox']
            #x, y, w, h = bbox
            #cv2.circle(img, (x, y), 5, (255, 0, 0), cv2.FILLED)
            #cv2.circle(img, (x + w, y + h), 5, (0, 0, 255), cv2.FILLED)
            #cartoon = cv2.resize(handMask, (w, h))
            #img = cvzone.overlayPNG(img, cartoon, (x, y))
            cartoonH = handMasks[h_th]
            mask_filter(cartoonH, "hand")
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('r'):
        f_th = 2
        h_th = 5