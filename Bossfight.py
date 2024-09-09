import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import random

#ตั้งกล่อง
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# เอารูปเข้า
imgBackground = cv2.imread("Resources/newBG.png")
imgGameOver = cv2.imread("Resources/gameOver.png")
imgBall = cv2.imread("Resources/purple.png", cv2.IMREAD_UNCHANGED)
imgBat1 = cv2.imread("Resources/batgojo.png", cv2.IMREAD_UNCHANGED)
imgBat2 = cv2.imread("Resources/batskn.png", cv2.IMREAD_UNCHANGED)




# ตัวตรวจจับมือ
detector = HandDetector(detectionCon=0.8, maxHands=2)




# ประกาศตัวแปรตอนเริ่มเกม ภายในเกมจะมีการเปลี่ยนค่า
ballPos = [100, 100]
speedX = 15
speedY = 15
gameOver = False
hp = [10, 10]
width = 200
height = 20
hp_bars = []
ball_scale = 1 + (10 - hp[1]) * 5  # ปรับค่า 0.1 เพื่อปรับความเร็วในการเปลี่ยนขนาด
height, width, _ = imgBall.shape
new_height = int(height * ball_scale)
new_width = int(width * ball_scale)
imgBallResized = cv2.resize(imgBall, (new_width, new_height))


#หลอดเลือด ทั้งสองฝ่าย
for i in range(11):  # 11 ภาพ ตั้งแต่ 0 ถึง 10
    hp_bars.append(cv2.imread(f"Resources/{i}Hpbar.png"))



while True:



    ball_scale = 1 + (10 - hp[1]) * 0.1  # ปรับค่า 0.1 เพื่อปรับความเร็วในการเปลี่ยนขนาด
    height, width, _ = imgBall.shape
    new_height = int(height * ball_scale)
    new_width = int(width * ball_scale)
    imgBallResized = cv2.resize(imgBall, (new_width, new_height))



    if cv2.waitKey(1) == ord('s'):
        speedX = 0
        speedY = 0


    _, img = cap.read()
    img = cv2.flip(img, 1)
    imgRaw = img.copy()

    # ตามหามือ
    hands, img = detector.findHands(img, flipType=False)

 # BACKGROUND
    img = cv2.addWeighted(img, 0, imgBackground, 0.8, 0)

    # เช็คมือ
    if hands:
        for hand in hands:
            x, y, w, h = hand['bbox']
            h1, w1, _ = imgBat1.shape
            y1 = y - h1 // 2
            y1 = np.clip(y1, 80, 415)
            #if มือข้างซ้าย
            if hand['type'] == "Left":
                img = cvzone.overlayPNG(img, imgBat1, (59, y1))
                if 59 < ballPos[0] < 59 + w1 and y1 < ballPos[1] < y1 + h1:   #ตรวจสอบว่าลูกบอลปะทะกับไม้ตีซ้ายหรือไม่ โดยเปรียบเทียบพิกัดของลูกบอล (ballPos) กับขอบของไม้ตี
                    speedX = -speedX
                    ballPos[0] += 30

    # Game Over
    if ballPos[0] < 40 or ballPos[0] > 1200:
        # ตรวจสอบว่าใครเป็นฝ่ายเสียแต้ม  ถ้าโดนชน = -1 HP
        if ballPos[0] < 40:
            hp[0] -= 1
        else:
            hp[1] -= 1

        # ตรวจสอบว่ามีใคร HP เป็น 0
        gameOver = hp[0] == 0 or hp[1] == 0

        # Reset ตำแหน่งบอลและกำหนดทิศทางใหม่ เมื่อมีคนโดนบอล อัดเข้าตัว
        if hp[0] != 0 or hp[1] != 0:
            ballPos = [640, 100]
            speedX = random.choice([-40, 40])  # กำหนดทิศทาง X แบบสุ่ม
            speedY = random.choice([-40, 40])  # กำหนดทิศทาง Y แบบสุ่ม
    # If game not over move the ball
    if hp[0] == 0 or hp[1] == 0:
        gameOver = True
        # ถ้าเกมจบ
        if gameOver:
            # แสดงข้อความ HP บนภาพ gameOver ตอนจบเกม
            for i in range(2):img = imgGameOver

    else:

        # Move the Ball
        if ballPos[1] >= 500 or ballPos[1] <= 10:
            speedY = -speedY

        ballPos[0] += speedX
        ballPos[1] += speedY

        # Draw the ball

        img = cvzone.overlayPNG(img, imgBallResized, ballPos)


        #HP BAR
        hp_bar_index = min(hp[0], 10)  # คำนวณ index ของภาพ HP bar
        x = 100  # ตำแหน่งของ HP bar ของ player 1
        y = 50
        img[y:y + hp_bars[hp_bar_index].shape[0], x:x + hp_bars[hp_bar_index].shape[1]] = hp_bars[hp_bar_index]

        # แสดง HP bar ของ player 2 เท่านั้น
        hp_bar_index = min(hp[1], 10)  # คำนวณ index ของภาพ HP bar
        x = 900  # ตำแหน่งของ HP bar ของ player 2
        y = 50
        img[y:y + hp_bars[hp_bar_index].shape[0], x:x + hp_bars[hp_bar_index].shape[1]] = hp_bars[hp_bar_index]

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord('r'):
        ballPos = [100, 100]
        speedX = 40
        speedY = 40
        gameOver = False
        hp = [10, 10]
        imgGameOver = cv2.imread("Resources/gameOver.png")