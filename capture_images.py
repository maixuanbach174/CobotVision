import cv2
from pathlib import Path

num = 500
folder = Path('dataset/action_' + str(num))
folder.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
frame = frame[254:444, 537:727]

while(True):
    cv2.imshow('img', frame)
    if cv2.waitKey(1) & 0xFF == ord('y'):
        cv2.imwrite(str(folder) + '/image.png', frame)
        cv2.destroyAllWindows()
        break
cap.release()