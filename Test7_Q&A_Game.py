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

# Initialize HandDetector class with the given parameters
detector = HandDetector(staticMode=False,
                        maxHands=1,
                        modelComplexity=1,
                        detectionCon=0.5,
                        minTrackCon=0.5)

# Read and resize background
BGimg_org = cv2.imread("GameResources/bg.jpg")
# resize background image to be the same size as image frame from camera
BGimg_org = cv2.resize(BGimg_org, (wf, hf))
blackBg = np.zeros([hf, wf, 3], dtype=np.uint8)

# Edit Question and Answer here for the choices in folder
questions = [["BEST STUDENT", "yuta"],
             ["BEST TEACHER", "yaha"],
             ["BEST SORCERER", "yuji"],
             ["STRONGEST SORCERER", "gojo"],
             ["Fire curse", "jogo"],
             ["Curse Doll", "panda"],
             ["Tengen medium", "gojo"],["Weakest Sorcerer", "momo"],["Best Brother", "todo"],["Best Healer", "shoko"]

            ]

# Create variable "imgObjects" to keep all object images
# Set initial values to them
pathFolderObject = "GameResources/choices"
imgObjects = []
choice_name = []
objectPoss = []
pathListObject = os.listdir(pathFolderObject)  # list of choice objects (png image) in folder
# print(pathListObject)
for path in pathListObject:
    object_filename = os.path.join(pathFolderObject, path)
    # print(object_filename)
    name = os.path.splitext(path)[0]
    imgObject = cv2.imread(object_filename, cv2.IMREAD_UNCHANGED)
    # print(imgObject.shape)
    if imgObject.shape[0] > 100:
        imgObject = cv2.resize(imgObject, ((imgObject.shape[1] * 100) // imgObject.shape[0], 100))  # (w,h)
    # print(imgObject.shape)
    imgObjects.append([imgObject, name])
    objectPoss.append([0, 2000])
def random_object():
    global img_choice
    num_object = len(imgObjects)
    x = range(30, wf - 150, 175)
    y = range(50, hf - 300, 100)
    x_pos = random.sample(x, k=len(x))
    y_pos = random.sample(y, k=len(y))
    x_pos.extend(x_pos)
    y_pos.extend(y_pos)
    y_pos.extend(y_pos)
    imgObjects_Appears = []
    img_choice = BGimg_org.copy()
    for i in range(num_object):
        imgObjects_Appear_name = imgObjects[i][1]
        imgObjects_Appear = imgObjects[i][0] # image of object
        scaling_size = random.uniform(0.9, 1)
        imgObjects_Appear = cv2.resize(imgObjects_Appear, (0, 0), fx=scaling_size, fy=scaling_size)
        objectPoss[i] = [x_pos[i], y_pos[i]]  # [x,y]
        # print(objectPoss[i])

        choice_area = blackBg.copy()
        imgObjects_Appear_mask = cv2.merge([imgObjects_Appear[:,:,3],imgObjects_Appear[:,:,3],imgObjects_Appear[:,:,3],imgObjects_Appear[:,:,3]])
        choice_area = cvzone.overlayPNG(choice_area, imgObjects_Appear_mask, objectPoss[i])  # (x,y)
        img_choice = cvzone.overlayPNG(img_choice, imgObjects_Appear, objectPoss[i])  # (x,y)
        imgObjects_Appears.append([choice_area, imgObjects_Appear_name])

    return img_choice, imgObjects_Appears

# Initial values
img_ratio = 0  # transparent level of camera frame (player)

img_choice, imgObjects_Appears = random_object()
questions_update = random.sample(questions, k = len(questions)) # random order of questions
selected_choice = None
choice_entry_times = {}
counter_answer = 0
current_question = 0
start_counter = False
answer_color = (0, 0, 255)
total_score = 0

def get_finger_location(img, imgWarped):
   """
   Get the location of the index fingertip.
   Parameters:
   - img: Original image.
   Returns:
   - finger_point: Coordinates of the index fingertip.
   """
   # Find hands in the current frame
   hands, img = detector.findHands(img, draw=False, flipType=True)
   # Check if any hands are detected
   if hands:
       # Information for the first hand detected
       hand1 = hands[0]  # Get the first hand detected
       finger_point = hand1["lmList"][8][0:2]  # List of 21 landmarks for the first hand
       cv2.circle(imgWarped, finger_point, 5, (255, 0, 0), cv2.FILLED)
   else:
       finger_point = None
   return finger_point

def create_overlay_image(imgObjects_Appears, warped_point, imgOverlay, imgOutput):
   """
   Create an overlay image based on the finger location.
   Parameters:
   - imgObjects_Appears: List of all choices appearing on the background .
   - warped_point: Coordinates of the index fingertip.
   - imgOverlay: Overlay image to be marked.
   - imgOutput: Output image of choices over background.
   Returns:
   - imgOverlay: Overlay image with marked color.
   - imgOutput: Output image of choices over background with masking counter.
   """
   choice_selected = None
   # Set the duration threshold for making a choice green
   green_duration_threshold = 1.0
   white = np.ones([hf, wf, 3], dtype=np.uint8)
   # loop through all the choices
   for mask, name in imgObjects_Appears:
       # print(mask[warped_point[1],warped_point[0],0])
       if mask[warped_point[1], warped_point[0], 0] == 255:
           print(name)
           # If the choice is not in the dictionary, add it with the current time
           if name not in choice_entry_times:
               choice_entry_times[name] = time.time()
           # Calculate the time the finger has spent in the choice
           time_in_choice = time.time() - choice_entry_times[name]
           # If the time is greater than the threshold, make the choice green
           if time_in_choice >= green_duration_threshold:
               color = (0, 255, 0)  # Green color
               choice_selected = name
           else:
               choice_selected = None
               color = (255, 0, 255)  # Magenta color
               # Draw an arc around the finger point based on elapsed time
               angle = int((time_in_choice / green_duration_threshold) * 360)
               cv2.ellipse(imgOutput, (warped_point[0] + 50, warped_point[1] - 50),
                           (25, 25), 0, 0, angle, (0, 0, 0),  # (50, 50)
                           thickness=-1)
               cv2.ellipse(imgOverlay, (warped_point[0] + 50, warped_point[1] - 50),
                           (25, 25), 0, 0, angle, (0, 255, 255),  # (50, 50)
                           thickness=-1)
           green_m = white*color
           green_m = green_m.astype(np.uint8)
           green_m = cv2.merge([green_m, mask])
           cvzone.overlayPNG(imgOverlay, green_m, (0,0))  # (x,y)
       else:
           # If the finger is not in the choice, remove it from the dictionary
           choice_entry_times.pop(name, None)
   return imgOverlay, choice_selected, imgOutput
def check_answer(name, current_question, img, total_score):
   global counter_answer, start_counter, answer_color
   if current_question == len(questions):
       cvzone.putTextRect(img, f"Your score is {total_score}/{len(questions)}", (0, 30), scale=2.5, thickness=3)
   if name != None and current_question < len(questions):
       if name == questions_update[current_question][1]:
           start_counter = 'CORRECT'
           answer_color = (0, 255, 0)
       else:
           start_counter = 'WRONG'
           answer_color = (0, 0, 255)
   if start_counter == False and name == None and current_question < len(questions):
       start_counter = 'NOT SELECT'
       answer_color = (0, 150, 150)
   if start_counter:
       counter_answer += 1
       if counter_answer != 0:
           cvzone.putTextRect(img, f"{start_counter}", (0, 710), scale=1.5, thickness=2, colorR=answer_color)
       if counter_answer == 70:
           counter_answer = 0
           if start_counter == "CORRECT":
               total_score += 1
           current_question += 1
           start_counter = False
   return current_question, total_score



while True:
    # Read a frame from the webcam
    success, img = cap.read()
    # print(img.shape)
    img = cv2.flip(img, 1)
    imgOutput = cv2.addWeighted(img, img_ratio, img_choice, 1 - img_ratio, 0)  # + 0
    finger_point = get_finger_location(img, imgOutput)

    # Create overlay image when select the choice
    imgOverlay = np.zeros((hf, wf, 3), dtype=np.uint8)
    selected_choice = None
    if finger_point and finger_point[1] < hf:
        imgOverlay, selected_choice, imgOutput = create_overlay_image(imgObjects_Appears, finger_point, imgOverlay,
                                                                      imgOutput)
        imgOutput = cv2.addWeighted(imgOutput, 1, imgOverlay, 0.65, 0)

    # Display the current question and check answer
    if current_question != len(questions):
        cvzone.putTextRect(imgOutput, questions_update[current_question][0], (0, 30), scale=2.5, thickness=3)
    current_question, total_score = check_answer(selected_choice, current_question, imgOutput, total_score)

    cv2.imshow("Q&A Game", imgOutput)


    key = cv2.waitKey(1)
    if key == ord('r'):
        choice_entry_times = {}
        current_question = 0
        total_score = 0
        counter_answer = 0
        start_counter = 'NOT SELECT'
        answer_color = (0, 150, 150)
        questions_update = random.sample(questions, k=len(questions))
        img_choice, imgObjects_Appears = random_object()
    elif key == 27:  # Esc
        break