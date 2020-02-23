import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'



model= load_model("./model.model")
cap=cv2.VideoCapture(0)

def preprocessandPredict(frame):
    
    frame=cv2.resize(frame,(28,28))
    frame=frame.reshape(-1,28,28,1)
    print(np.argmax(model.predict(frame)))
    return (np.argmax(model.predict(frame)),np.max(model.predict(frame)))
    
    
    



while(True):
    ret,frame=cap.read()
    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    frame = cv2.GaussianBlur(frame,(9,9),0)
    frame=-frame/255.0
    frame[frame < 0.4] = 1.0
    predic=preprocessandPredict(frame)
    if predic[1]>0.1:
        frame=cv2.putText(frame,str(predic[0]),(100,100),cv2.FONT_HERSHEY_SIMPLEX,1.5,(255,255,255))
    cv2.imshow("Img",frame)
    if cv2.waitKey(1)== ord('q'):
        break   
cv2.destroyAllWindows()