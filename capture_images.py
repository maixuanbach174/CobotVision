import cv2

cap = cv2.VideoCapture(0)
ret, frame = cap.read()

while(True):
    cv2.imshow('img', frame)
    if cv2.waitKey(1) & 0xFF == ord('y'):
        cv2.imwrite('dataset/action1/image.png', frame)
        cv2.destroyAllWindows()
        break
cap.release()