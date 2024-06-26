import torch
import cv2
import numpy as np
from mss import mss



model = torch.hub.load(r'C:\Users\Admin\바탕화면\asd\yolov5-master', 'custom', path=r'C:\Users\Admin\바탕화면\asd\valorant-10.pt', source='local')

with mss() as sct:
    monitor = {"top": 220, "left": 640, "width": 640, "height":640}

    while(True):
        screenshot = np.array(sct.grab(monitor))
        results = model(screenshot, size=640)
        df= results.pandas().xyxy[0]
        try:
            xmin = int(df.iloc[0,0])
            ymin = int(df.iloc[0,1])
            xmax = int(df.iloc[0,2])
            ymax = int(df.iloc[0,3])
            
            cv2.rectangle(screenshot, (xmin, ymin), (xmax, ymax), (255,0,0), 2)
        except:
            print("",end="")

        cv2.imshow("frame", screenshot)
        if(cv2.waitKey(1) == ord('q')):
            cv2.destroyAllWindows()
            break