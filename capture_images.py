import cv2
from pathlib import Path

num = 'dataset/action_report/image20.png'


cap = cv2.VideoCapture(0)
ret, frame = cap.read()
frame = frame[254:444, 537:727]

while(True):
    cv2.imshow('img', frame)
    if cv2.waitKey(1) & 0xFF == ord('y'):
        cv2.imwrite(num, frame)
        cv2.destroyAllWindows()
        break
cap.release()