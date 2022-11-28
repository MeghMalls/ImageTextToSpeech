import cv2
import pytesseract
from PIL import Image

#pytesseract

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
img = cv2.imread('DemoText.png')
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
