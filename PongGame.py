import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import random

# ตั้งกล่อง
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# เอารูปเข้า
imgBackground = cv2.imread("Resources/newBG 2.png")
imgGameOver = cv2.imread("Resources/game.png")
imgBall = cv2.imread("Resources/purple.png", cv2.IMREAD_UNCHANGED)
imgBat1 = cv2.imread("Resources/batgojo.png", cv2.IMREAD_UNCHANGED)
imgBat2 = cv2.imread("Resources/batskn.png", cv2.IMREAD_UNCHANGED)

# ตัวตรวจจับมือ
detector = HandDetector(detectionCon=0.8, maxHands=2)

# ประกาศตัวแปรตอนเริ่มเกม
ballPos = [100, 100]
speedX = 15
speedY = 15
gameOver = False
hp = [10, 100]
width = 200
height = 20
hp_bars = []
min_speed = 5
max_speed = 40

for i in range(11):  # 11 ภาพ ตั้งแต่ 0 ถึง 10
    hp_bars.append(cv2.imread(f"Resources/{i}Hpbar.png"))


def create_ball(min_size, max_size):
    ball_scale = random.uniform(min_size, max_size)
    return imgBall


create_ball(0.5, 2)

while True:
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

            # วางแป้นสำหรับมือซ้าย
            if hand['type'] == "Left":
                img = cvzone.overlayPNG(img, imgBat1, (59, y1))
                if 59 < ballPos[0] < 59 + w1 and y1 < ballPos[1] < y1 + h1:
                    speedX = -speedX
                    speedY = random.randint(min_speed, max_speed)



    # Game Over
    if ballPos[0] < 40 or ballPos[0] > 1200:
        if ballPos[0] < 40:
            hp[0] -= 1
        else:
            hp[1] -= 1

        gameOver = hp[0] == 0 or hp[1] == 0
        if hp[0] != 0 or hp[1] != 0:
            ballPos = [400, 100]
            speedX = random.choice([-40, 40])
            speedY = random.choice([-40, 40])
            create_ball(0.5, 2)

    if hp[0] == 0 or hp[1] == 0:
        gameOver = True
        if gameOver:
            for i in range(2):
                cv2.putText(imgGameOver, f"PLAYER {i + 1} HP = {hp[i]}", (100, 200 + 100 * i), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 255, 255), 2)
            img = imgGameOver
    else:
        if ballPos[1] >= 500 or ballPos[1] <= 10:
            speedY = -speedY

        ballPos[0] += speedX
        ballPos[1] += speedY

        # วาดลูกบอล
        img = cvzone.overlayPNG(img, imgBall, ballPos)

        # HP BAR
        hp_bar_index = min(hp[0], 10)
        img[50:50 + hp_bars[hp_bar_index].shape[0], 100:100 + hp_bars[hp_bar_index].shape[1]] = hp_bars[hp_bar_index]

        hp_bar_index = min(max(hp[1] // 10, 0), 10)
        img[50:50 + hp_bars[hp_bar_index].shape[0], 900:900 + hp_bars[hp_bar_index].shape[1]] = hp_bars[hp_bar_index]

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord('r'):
        ballPos = [100, 100]
        speedX = 40
        speedY = 40
        gameOver = False
        hp = [10, 100]
        imgGameOver = cv2.imread("Resources/gameOver.png")
