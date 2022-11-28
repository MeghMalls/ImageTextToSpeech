import cv2
import pytesseract
from PIL import Image

#PILLOW and OpenCV

camera = cv2.VideoCapture(0)

while True:
    _,Image=camera.read()
    cv2.imshow('Text Detection', Image)
    if cv2.waitKey(1)& 0xFF==ord('s'):
        cv2.imwrite('test1.png',Image)
        break
camera.release()
cv2.destroyAllWindows()

#pytesseract

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
img = cv2.imread('test1.png')
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
text=pytesseract.image_to_string(img)
print(text)
cv2.imshow('Result',img)


#gTTS

from gtts import gTTS
import os

language ='en'
output =gTTS(text=text, lang=language, slow= False)
output.save("output.mp3")
os.system("start output.mp3")
cv2.waitKey(0)
