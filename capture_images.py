import cv2
from pathlib import Path

num = 'report'
my_folder = Path("dataset/action_" + num)
print(my_folder)
my_folder.mkdir(exist_ok=True, parents=True)

cap = cv2.VideoCapture(0)
ret, frame = cap.read()

while(True):
    cv2.imshow('img', frame)
    if cv2.waitKey(1) & 0xFF == ord('y'):
        cv2.imwrite(str(my_folder) + '/image.png', frame)
        cv2.destroyAllWindows()
        break
cap.release()