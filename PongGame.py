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
imgBackground = cv2.imread("Resources/BGelden.png")
imgGameOver = cv2.imread("Resources/gameover.jpg")
imgWin = cv2.imread("Resources/win.jpg")  # ภาพที่จะแสดงเมื่อผู้เล่น 2 ชนะ
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
max_speed = 200
bat_pos_left = (59, 0)  # ตำแหน่งของแป้นซ้ายเริ่มต้น
ballMoving = True  # ตัวแปรที่ใช้ตรวจสอบสถานะการเคลื่อนไหวของบอล

for i in range(11):  # 11 ภาพ ตั้งแต่ 0 ถึง 10
    hp_bars.append(cv2.imread(f"Resources/{i}Hpbar.png"))

# ฟังก์ชันสร้างบอลพร้อมขนาดสุ่ม
def create_ball(min_size, max_size):
    ball_scale = random.uniform(min_size, max_size)
    new_ball = cv2.resize(imgBall, (0, 0), fx=ball_scale, fy=ball_scale)
    return new_ball

# ฟังก์ชันสุ่มความเร็วบอล
def randomize_ball_speed():
    return random.randint(min_speed, max_speed), random.randint(min_speed, max_speed)

# ตั้งค่าเวลาเริ่มต้นสำหรับสุ่มขนาดบอลและความเร็วบอล
start_time_size = cv2.getTickCount()
start_time_speed = cv2.getTickCount()
ball_change_interval = 5  # หน่วยเป็นวินาที
imgBallResized = create_ball(0.5, 2)  # ขนาดบอลเริ่มต้น
speedX, speedY = randomize_ball_speed()

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
        ballMoving = True  # บอลเคลื่อนไหวได้เมื่อพบมือ
        for hand in hands:
            x, y, w, h = hand['bbox']
            h1, w1, _ = imgBat1.shape
            x1 = x - w1 // 2
            y1 = y - h1 // 2
            x1 = np.clip(x1, 0, 1280 - w1)  # ตรวจสอบการขยับแป้นไม่ให้เกินขอบจอ
            y1 = np.clip(y1, 0, 720 - h1)  # ตรวจสอบการขยับแป้นไม่ให้เกินขอบจอ

            # วางแป้นสำหรับมือซ้าย
            if hand['type'] == "Right":
                img = cvzone.overlayPNG(img, imgBat1, (x1, y1))
                bat_pos_left = (x1, y1)  # อัปเดตตำแหน่งแป้นซ้าย

                # เช็คการชนกับลูกบอล
                if bat_pos_left[0] < ballPos[0] < bat_pos_left[0] + w1 and bat_pos_left[1] < ballPos[1] < bat_pos_left[1] + h1:
                    speedX = -speedX
                    speedY = random.randint(min_speed, max_speed)

    else:
        ballMoving = False  # บอลไม่เคลื่อนไหวเมื่อไม่พบมือ

    # เปลี่ยนขนาดบอลทุก 5 วินาที
    current_time_size = (cv2.getTickCount() - start_time_size) / cv2.getTickFrequency()
    if current_time_size > ball_change_interval:
        imgBallResized = create_ball(0.5, 2)  # สุ่มขนาดบอลใหม่
        start_time_size = cv2.getTickCount()  # รีเซ็ตเวลาเริ่มต้น

    # สุ่มความเร็วบอลทุก 5 วินาที
    current_time_speed = (cv2.getTickCount() - start_time_speed) / cv2.getTickFrequency()
    if current_time_speed > ball_change_interval:
        speedX, speedY = randomize_ball_speed()  # สุ่มความเร็วบอลใหม่
        start_time_speed = cv2.getTickCount()  # รีเซ็ตเวลาเริ่มต้น

    # ตรวจสอบบอลหลุดจอ
    if ballPos[0] < -50 or ballPos[0] > 1330 or ballPos[1] < -50 or ballPos[1] > 770:
        if ballPos[0] < -50:
            hp[0] -= 1
        if ballPos[0] > 1330:
            hp[1] -= 1

        gameOver = hp[0] == 0 or hp[1] == 0
        if hp[0] != 0 or hp[1] != 0:
            ballPos = [bat_pos_left[0] + w1 // 2, bat_pos_left[1] - 10]  # วางบอลที่แป้น
            speedX, speedY = randomize_ball_speed()  # สุ่มความเร็วบอลใหม่
            imgBallResized = create_ball(0.5, 2)  # สุ่มขนาดบอลใหม่

    if hp[0] == 0:
        gameOver = True
        img = imgGameOver  # แสดงภาพ gameover
    elif hp[1] == 0:
        gameOver = True
        img = imgWin  # แสดงภาพ win
    else:
        # ตรวจสอบการชนกับขอบบนและขอบล่าง
        if ballPos[1] <= 0 or ballPos[1] >= 720 - imgBallResized.shape[0]:
            speedY = -speedY

        if ballMoving:  # เคลื่อนไหวบอลเมื่อ ballMoving เป็น True
            ballPos[0] += speedX
            ballPos[1] += speedY

        # วาดลูกบอล
        img = cvzone.overlayPNG(img, imgBallResized, ballPos)

        # HP BAR
        hp_bar_index = min(hp[0], 10)
        img[50:50 + hp_bars[hp_bar_index].shape[0], 100:100 + hp_bars[hp_bar_index].shape[1]] = hp_bars[hp_bar_index]

        hp_bar_index = min(max(hp[1] // 10, 0), 10)
        img[50:50 + hp_bars[hp_bar_index].shape[0], 900:900 + hp_bars[hp_bar_index].shape[1]] = hp_bars[hp_bar_index]

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord('r'):
        ballPos = [bat_pos_left[0] + w1 // 2, bat_pos_left[1] - 10]  # วางบอลที่แป้น
        speedX, speedY = randomize_ball_speed()
        gameOver = False
        hp = [10, 100]
        imgGameOver = cv2.imread("Resources/gameover.jpg")  # โหลดภาพ gameover ใหม่
        imgWin = cv2.imread("Resources/win.jpg")  # โหลดภาพ win ใหม่
