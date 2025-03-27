import cv2
import numpy as np
import mediapipe as m

from math import sqrt
import pyautogui
import time

# Initialize MediaPipe hands and drawing utils
m_drawing = m.solutions.drawing_utils
m_hands = m.solutions.hands

click = 0
double_click = 0
scroll_y = 0
last_click_time = time.time()
double_click_threshold = 0.3  # Seconds

video = cv2.VideoCapture(0)
with m_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8) as hands:
    while video.isOpened():
        _, frame = video.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)
        imageHeight, imageWidth, _ = image.shape
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                m_drawing.draw_landmarks(image, hand, m_hands.HAND_CONNECTIONS,
                                         m_drawing.DrawingSpec(color=(250, 0, 0), thickness=2, circle_radius=2))

                # Extract landmark positions
                index_finger_tip = hand.landmark[m_hands.HandLandmark.INDEX_FINGER_TIP]
                thumb_tip = hand.landmark[m_hands.HandLandmark.THUMB_TIP]

                # Convert normalized coordinates to pixel coordinates
                index_finger_tip_x = int(index_finger_tip.x * imageWidth)
                index_finger_tip_y = int(index_finger_tip.y * imageHeight)
                thumb_tip_x = int(thumb_tip.x * imageWidth)
                thumb_tip_y = int(thumb_tip.y * imageHeight)

                # Move the cursor
                screen_width, screen_height = pyautogui.size()
                cursor_x = int(screen_width * index_finger_tip.x)
                cursor_y = int(screen_height * index_finger_tip.y)
                pyautogui.moveTo(cursor_x, cursor_y)

                # Calculate the distance between the index finger tip and thumb tip
                distance = sqrt((index_finger_tip_x - thumb_tip_x) ** 2 + (index_finger_tip_y - thumb_tip_y) ** 2)

                # Perform a click if the distance is below a certain threshold
                if distance < 20:
                    current_time = time.time()
                    if current_time - last_click_time < double_click_threshold:
                        double_click += 1
                        if double_click % 2 == 0:
                            print("Double click")
                            pyautogui.doubleClick()
                    else:
                        click += 1
                        if click % 5 == 0:
                            print("Single click")
                            pyautogui.click()
                    last_click_time = current_time

                # Check for scrolling action
                if distance < 20:
                    scroll_y += index_finger_tip_y - thumb_tip_y
                    if abs(scroll_y) > 30:
                        if scroll_y > 0:
                            pyautogui.scroll(-1)  # Scroll down
                        else:
                            pyautogui.scroll(1)  # Scroll up
                        scroll_y = 0

        # Display the image with landmarks
        cv2.imshow('Virtual Mouse', image)

        # Exit the loop when 'x' key is pressed
        if cv2.waitKey(10) & 0xFF == ord('x'):
            break

# Release video capture and close OpenCV window
video.release()
cv2.destroyAllWindows()
