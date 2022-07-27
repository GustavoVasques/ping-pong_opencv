import cv2 as cv
import mediapipe as mp
import time


webcam = cv.VideoCapture(0)
webcam.set(3, 1280)
webcam.set(4, 720)

imgBackground = cv.imread("Resources/Background.png")
imgGameOver = cv.imread("Resources/gameOver.png")
imgBall = cv.imread("Resources/Ball.png", cv.IMREAD_UNCHANGED)
imgBat1 = cv.imread("Resources/bat1.png", cv.IMREAD_UNCHANGED)
imgBat2 = cv.imread("Resources/bat2.png", cv.IMREAD_UNCHANGED)

mphands = mp.solutions.hands

hands = mphands.Hands(static_image_mode=False,
                      max_num_hands=2,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)

mpDraw = mp.solutions.drawing_utils

circle_x = 1280 // 2
circle_y = 720 // 2
speed_x = 3
speed_y = 3

time.sleep(2)
while True:
    lixo, img = webcam.read()

    cv.circle(img, (circle_x, circle_y), 20, (0, 0, 255), 40)

    img = cv.flip(img, 1)

    img = cv.addWeighted(img, 0.5, imgBackground, 0.5, 0)
    h, w, c = img.shape

    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    coordinates = hands.process(img_rgb)
    hand_landmarks = coordinates.multi_hand_landmarks

    if hand_landmarks:
        for handLMs in hand_landmarks:
            x_max = 0
            y_max = 0
            x_min = w
            y_min = h
            print(hand_landmarks)

            for lm in handLMs.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y

                y_mid = (y_max + y_min)//2

            if handLMs == hand_landmarks[0]:
                cv.line(img, (59, y_mid),
                        (59, y_mid - 100), (0, 255, 0), 50)

            else:
                cv.line(img, (1195, y_mid),
                        (1195, y_mid - 100), (255, 0, 0), 50)

            if 59 <= circle_x <= 109 and (y_mid - 100) <= circle_y <= y_mid:
                speed_x = -speed_x
                speed_y = -speed_y

            if 1145 <= circle_x <= 1195 and (y_mid - 100) <= circle_y <= y_mid:
                speed_x = -speed_x
                speed_y = -speed_y

            cv.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    if circle_y <= 50 or circle_y >= 500:
        speed_y = -speed_y

    circle_x += speed_x
    circle_y += speed_y

    cv.imshow('webcam', img)
    cv.waitKey(1)
