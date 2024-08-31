
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
imgBackground = cv2.imread("Resources/Background.png")
imgGameOver = cv2.imread("Resources/gameOver.png")
imgBall = cv2.imread("Resources/Ball.png", cv2.IMREAD_UNCHANGED)
imgBat1 = cv2.imread("Resources/bat1.png", cv2.IMREAD_UNCHANGED)
imgBat2 = cv2.imread("Resources/bat2.png", cv2.IMREAD_UNCHANGED)

# ตัวตรวจจับมือ
detector = HandDetector(detectionCon=0.8, maxHands=2)




# ประกาศตัวแปรตอนเริ่มเกม ภายในเกมจะมีการเปลี่ยนค่า
ballPos = [100, 100]
speedX = 40
speedY = 40
gameOver = False
# เปลี่ยนจาก score เป็น hp
hp = [10, 10]
width = 200
height = 20
hp_bars = []

for i in range(11):  # 11 ภาพ ตั้งแต่ 0 ถึง 10
    hp_bars.append(cv2.imread(f"Resources/{i}Hpbar.png"))



while True:


    _, img = cap.read()
    img = cv2.flip(img, 1)
    imgRaw = img.copy()

    # ตามหามือ
    hands, img = detector.findHands(img, flipType=False)

    # ปรับค่าความโปรงใสของภาพ
    img = cv2.addWeighted(img, 0.2, imgBackground, 0.8, 0)

    # เช็คมือ
    if hands:
        for hand in hands:
            x, y, w, h = hand['bbox']
            h1, w1, _ = imgBat1.shape
            y1 = y - h1 // 2
            y1 = np.clip(y1, 20, 415)
            #if มือข้างซ้าย
            if hand['type'] == "Left":
                img = cvzone.overlayPNG(img, imgBat1, (59, y1))
                if 59 < ballPos[0] < 59 + w1 and y1 < ballPos[1] < y1 + h1:
                    speedX = -speedX
                    ballPos[0] += 30


            if hand['type'] == "Right":
                img = cvzone.overlayPNG(img, imgBat2, (1195, y1))
                if 1195 - 50 < ballPos[0] < 1195 and y1 < ballPos[1] < y1 + h1:
                    speedX = -speedX
                    ballPos[0] -= 30







    # Game Over
    if ballPos[0] < 40 or ballPos[0] > 1200:
        # ตรวจสอบว่าใครเป็นฝ่ายเสียแต้ม  ถ้าโดนชน = -1 HP
        if ballPos[0] < 40:
            hp[0] -= 1
        else:
            hp[1] -= 1

        # ตรวจสอบว่ามีใคร HP เป็น 0
        gameOver = hp[0] == 0 or hp[1] == 0

        # Reset ตำแหน่งบอลและกำหนดทิศทางใหม่
        if hp[0] != 0 or hp[1] != 0:
            ballPos = [400, 100]
            speedX = random.choice([-40, 40])  # กำหนดทิศทาง X แบบสุ่ม
            speedY = random.choice([-40, 40])  # กำหนดทิศทาง Y แบบสุ่ม
    # If game not over move the ball
    if hp[0] == 0 or hp[1] == 0:
        gameOver = True
        # ถ้าเกมจบ
        if gameOver:
            # แสดงข้อความ HP บนภาพ gameOver ตอนจบเกม
            for i in range(2):
                cv2.putText(imgGameOver, f"PLAYER {i+1} HP = {hp[i]}", (100, 200 + 100 * i), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 255, 255), 2)
            img = imgGameOver

    else:

        # Move the Ball
        if ballPos[1] >= 500 or ballPos[1] <= 10:
            speedY = -speedY

        ballPos[0] += speedX
        ballPos[1] += speedY

        # Draw the ball
        img = cvzone.overlayPNG(img, imgBall, ballPos)

        for i in range(2):
            hp_bar_index = min(hp[i], 10)  # คำนวณ index ของภาพ HP bar
            # กำหนดตำแหน่งของภาพ HP bar
            x = 300 if i == 0 else 900
            y = 600
            # แสดงภาพ HP bar ทับลงบนภาพหลัก
            img[y:y + hp_bars[hp_bar_index].shape[0], x:x + hp_bars[hp_bar_index].shape[1]] = hp_bars[hp_bar_index]

    img[580:700, 20:233] = cv2.resize(imgRaw, (213, 120))

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord('r'):
        ballPos = [100, 100]
        speedX = 40
        speedY = 40
        gameOver = False
        hp = [10, 10]
        imgGameOver = cv2.imread("Resources/gameOver.png")