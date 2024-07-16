import cv2
from cvzone.PoseModule import PoseDetector
# Initialize the webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# Initialize the PoseDetector class with the given parameters
detector = PoseDetector(staticMode=False,
                          modelComplexity=1,
                          smoothLandmarks=True,
                          enableSegmentation=False,
                          smoothSegmentation=True,
                          detectionCon=0.5,
                          trackCon=0.5)

previous_angle=0
count=0

while True:
 # Capture each frame from the webcam
 success, img = cap.read()
 # Find the human pose in the frame
 img = detector.findPose(img)
 # Find the landmarks, bounding box, and center of the body in the frame
 # Set draw=True to draw the landmarks and bounding box on the image
 lmList, bboxInfo = detector.findPosition(img, draw=True, bboxWithHands=False)
 # print(bboxInfo)
 # print(lmList)
 # Display the frame in a window

 # Check if any body landmarks are detected
 if lmList:
      # Get the center of the bounding box around the body
      center = bboxInfo["center"]
      # Draw a circle at the center of the bounding box
      cv2.circle(img, center, 5, (255, 0, 255), cv2.FILLED)
      # Calculate the distance between landmarks 11 and 15 and draw it on the image
      length, img, info = detector.findDistance(lmList[24][0:2],
                                                lmList[28][0:2],
                                                img=img,
                                                color=(255, 0, 0),
                                                scale=10)

      print(info)
      cv2.circle(img, info[0:2], 5, (255, 0, 0), cv2.FILLED)  # b
      cv2.circle(img, info[2:4], 5, (0, 255, 0), cv2.FILLED)  # g
      cv2.circle(img, info[4:6], 5, (0, 0, 255), cv2.FILLED)  # r
      print(info[4:6])
      print(length)
      x2, y2 = info[4:6]
      cv2.putText(img, str(int(length)), (x2, y2), cv2.FONT_HERSHEY_PLAIN,
                  2, (255, 0, 0), 3, cv2.LINE_AA)
      cv2.putText(img, str(int(length)), (x2, y2), cv2.FONT_HERSHEY_PLAIN,
                  2, (0, 255, 255), 1, cv2.LINE_AA)
      angle, img = detector.findAngle(lmList[24][0:2],
                                      lmList[26][0:2],
                                      lmList[28][0:2],
                                      img=img,
                                      color=(0, 0, 255),
                                      scale=10)
      # Check if the angle is less than 30 degrees
      print(angle)
      if angle <= 30 and previous_angle > 30:
           count += 1
      previous_angle = angle

      print(count)
      cv2.putText(img, str(count), (10, 70),
               cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 255), 5)




 cv2.imshow("Image", img)
 # Check for 'q' key press to break the loop and close the window

 if cv2.waitKey(1) & 0xFF == ord('q'):
     break