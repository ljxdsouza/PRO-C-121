import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

finger_tips = [8, 12, 16, 20]
thumb_tip = 4

while True:
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    h, w, c = img.shape
    results = hands.process(img)

    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            # Accessing the landmarks by their position
            lm_list = []
            for id, lm in enumerate(hand_landmark.landmark):
                lm_list.append(lm)

            # Get the positions of finger tips and thumb
            finger_tip_positions = [(int(lm_list[i].x * w), int(lm_list[i].y * h)) for i in finger_tips]
            thumb_position = (int(lm_list[thumb_tip].x * w), int(lm_list[thumb_tip].y * h))

            # Draw circles around finger tips
            for pos in finger_tip_positions:
                cv2.circle(img, pos, 8, (255, 0, 0), cv2.FILLED)

            # Check if fingers are folded
            finger_fold_status = []
            for i in range(len(finger_tip_positions) - 1):
                if finger_tip_positions[i][0] < finger_tip_positions[i + 1][0]:
                    finger_fold_status.append(True)
                else:
                    finger_fold_status.append(False)

            # Check if thumb is raised up or down
            if thumb_position[1] < finger_tip_positions[0][1]:
                print("LIKE")
                cv2.putText(img, "LIKE", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                print("DISLIKE")
                cv2.putText(img, "DISLIKE", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            mp_draw.draw_landmarks(img, hand_landmark,
                                   mp_hands.HAND_CONNECTIONS, mp_draw.DrawingSpec((0, 0, 255), 2, 2),
                                   mp_draw.DrawingSpec((0, 255, 0), 4, 2))

    cv2.imshow("hand tracking", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
