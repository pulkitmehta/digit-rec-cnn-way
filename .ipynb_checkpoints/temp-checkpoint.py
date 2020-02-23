import cv2
cap=cv2.VideoCapture(0)

while(True):
    ret,frame=cap.read()
    frame[frame < 128] = 0
    cv2.imshow("Img",frame)
    if cv2.waitKey(1)== ord('q'):
        break   
cv2.destroyAllWindows()