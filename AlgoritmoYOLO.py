import cv2
from darkflow.net.build import TFNet
import numpy as np
import time

option = {
    'model': 'cfg/yolo.cfg',
    'load': 'bin/yolov2.weights',
    'threshold': 0.7,
    'gpu': 1.0
}

tfnet = TFNet(option)
t_s=0
t_b=0

capture = cv2.VideoCapture('C:/Users/karee/anaconda3/darkflow/videoA3.mp4')
#capture = cv2.VideoCapture('C:/Users/karee/anaconda3/darkflow/video1.mp4')
#capture = cv2.VideoCapture('C:/Users/karee/anaconda3/darkflow/videoA2.mp4')
colors = [tuple(255 * np.random.rand(3)) for i in range(5)]

while (capture.isOpened()):
    stime = time.time()
    ret, frame = capture.read()
    if ret:
        results = tfnet.return_predict(frame)
        for color, result in zip(colors, results):
            confidence = (result['confidence'])
            tl = (result['topleft']['x'], result['topleft']['y'])
            br = (result['bottomright']['x'], result['bottomright']['y'])
            label = result['label']
            text = '{}: {:.0f}%'.format(label, confidence * 100)
            frame = cv2.rectangle(frame, tl, br, color, 7)
            frame = cv2.putText(frame, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)

            #cv2.line(frame,(0,150),(800,150),(0,0,255),2) #Video 1
            #cv2.line(frame,(0,100),(800,100),(0,0,255),2) #Video 2
            cv2.line(frame,(0,220),(800,220),(0,0,255),2) #Video 3
            
            #punto medio
            inicioX = (result['topleft']['x'])
            inicioY = (result['topleft']['y'])
            finX = (result['bottomright']['x'])
            finY = (result['bottomright']['y'])
            puntoMedioX = (inicioX+finX)/2
            puntoMedioY = (inicioY+finY)/2
            x = int(puntoMedioX)
            y = int(puntoMedioY)
            posicion = inicioX - finY
            #circulo
            cv2.circle (frame, (x,y),1,(0,0,255), 2)
            #if y >= 148 and y <= 152: #Video 1
            #if y >= 98 and y <= 105: #video 2
            if y >= 208 and y <= 220: #video3
                if(posicion>0):
                    posicionR='entrada'
                    t_b = t_b+1
                else:
                    posicionR='salida'
                    t_s = t_s+1
                #if CLASSES[idx] == "person":
                #f = open("conteo_V3.txt", "r")
                f = open("conteoprueba.txt", "r")
                file_lines = f.readlines()
                f.close()
                ultima_linea = file_lines[len(file_lines)-1].split('|')
                print(ultima_linea[0])
                #g = open("conteo_V3.txt" , "a")
                g = open("conteoprueba.txt" , "a")
                escribir = int(ultima_linea[0])+1
                #g.write(escribir)
                g.write("%d|%s|%s\n" % (escribir,label,posicionR))
                g.close()

        # show the output frame
        cv2.imshow('frame', frame)
        print('FPS {:.1f}'.format(1 / (time.time() - stime)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
  
    else:
        # do a bit of cleanup
        capture.release()
        cv2.destroyAllWindows()
        break