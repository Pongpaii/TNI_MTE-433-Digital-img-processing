import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np

color_level=255
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# Set the frame width to 640 pixels
cap.set(3, 640)
# Set the frame height to 480 pixels
cap.set(4, 480)
BGimg_org = cv2.imread("resource/meme.jpg")
# resize the background image to be the same size as image frame from camera
BGimg_org = cv2.resize(BGimg_org, (640, 480))
imgPNG = cv2.imread("resource/meme.jpg",cv2.IMREAD_UNCHANGED)
imgPNG = cv2.resize(imgPNG, (0, 0), None, 0.03, 0.03)
# Initialize the SelfiSegmentation class. It will be used for background removal.
# model is 0 or 1 : 0 is general, 1 is landscape(faster)
segmentor = SelfiSegmentation(model=0)
detectorH = HandDetector(detectionCon=0.8, maxHands=1)


# Infinite loop to keep capturing frames from the webcam
while True:
     # Capture a single frame
     success, img = cap.read()
     # Use the SelfiSegmentation class to remove the background
     # imgBG can be a color or an image(same size as the original image)
     # 'cutThreshold' is the sensitivity of the segmentation (higher = more cut, lower = less cut)
     #imgOut = segmentor.removeBG(img, imgBg=(0, 255, 255), cutThreshold=0.5)
     #imgOut = cvzone.overlayPNG(imgOut, imgPNG, pos=[10, 10])

     imgOut = segmentor.removeBG(img, imgBg=(254, 243, 255 - color_level), cutThreshold=0.5)
     hands, img = detectorH.findHands(img, draw=True)
     if hands:
         hand = hands[0]
     lmList = hand["lmList"]  # List of 21 Landmark points
     if lmList[8][1] > 0:  # y of index fingertip
         y = lmList[8][1]
     color_level = y * (255 / 480)
     color_level = int(np.clip(color_level, 0, 255))
     # print(color_level)




     # Stack the original image and the image with background removed side by side
     imgStacked = cvzone.stackImages([img, imgOut], cols=2, scale=1)
     # Display the stacked images
     cv2.imshow("Image", imgStacked)
     # Check for 'q' key press to break the loop and close the window
     if cv2.waitKey(1) & 0xFF == ord('q'):
        break


