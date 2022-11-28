# ImageTextToSpeech
A python project to detect Text from images and then convert the Text to Speech

Project:
Text Detection from Image File and Text-to-Speech Conversion

Aim:
The aim of the project is to create an interface that takes an image, processes it, recognizes and extracts text from it and converts it into a speech.

Objective:
The main objective that is being focused on is to build a tool that helps visually impaired people to hear the text around them (in the form of banners, signs, hoardings, posters, headlines etc.) that they are unable to see.

Technologies:
Some of the many technologies that were used in this project are:
	Image Processing
	Feature Extraction
	Optical Character Recognition
	Text Classification
	Machine Translation of Speech

	Machine Learning
	Neural Networks

Libraries, Packages and Tools:
Python was the chosen coding language for this project as python encompasses a wide range of libraries that can be integrated into our project to make it more efficient and easier to create, understand and manage. Some of the libraries used are:
•	PILLOW
•	OpenCV
•	Tesseract
•	gTTS


PILLOW    (pip install --upgrade Pillow)

-	Stands for Python Imaging Library
-	Open-Source Library that contains basic Image Processing functionality
-	Provides support for opening, manipulating, and saving many different image file formats

OpenCV    (pip install opencv-python)
-	Open-Source Computer Vision Library
-	Used for Image Processing
-	Has additional features such as Video Processing, Capturing image from Camera, Feature Extraction
-	Is implemented using cv2 and Numpy

Tesseract (and Pytesseract)      (pip install pytesseract)
-	It is an Optical Character Recognition (OCR) tool by Google
-	It is an OCR engine
-	Transforms a two-dimensional image of text from its image representation into machine-readable text
-	Performs functions such as Image Processing, Text Localization, Character Segmentation and Character Recognition 
The Tesseract tool can be implemented in Python by using ‘pytesseract’. Pytesseract or Python Tesseract acts as a wrapper for the Tesseract-OCR engine for its easy implementation in Python, that is, pytesseract is an interface for implementing tesseract using python.
        
gTTS    (pip install gTTS)
-	Stands for Google-Text-To-Speech
-	It is a python library that interfaces with Google Translate’s text-to-speech API.
-	It is used to convert a given string of text into an audio file (.mp3) format
-	It supports English along with other Foreign as well as Local languages

Prerequisites:
•	WebCam (or other image capturing device)
•	Python IDE ( Used here: PyCharm)
•	Download Tesseract to your system (tesseract-ocr-w64-setup-v5.2.0.20220712.exe)- link to download windows version.
-	Once downloaded, Run the .exe file to install Tesseract 
-	Next, go to the download location and copy the path of the location where the pytesseract’s ‘.exe’ file is saved
-	Now add this path to the system’s environment variables
•	Install the necessary libraries in the python interpreter (The following pip install commands can be used in the terminal):
-	PILLOW   ……………….. (pip install --upgrade Pillow)
-	OpenCV   ………………..(pip install opencv-python)
-	Pytesseract ……………..(pip install pytesseract)
-	gTTS    ……………………..(pip install gTTS)
•	Have an image file (preferably with clear text) saved in the project file.
(It is saved here as DemoText.png , it can be named differently but the name of the image file should be same as the name of the initializing image mentioned in the code)

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

CODE:

1) Use a pre-saved Image File to extract text from (main2.py)

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

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

2) Capture an Image from a live camera (webcam) and use the captured image to extract text from(main.py)

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

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

•	This Use-Case of the project is more prone to errors and bugs.
-Either the text recognition from the captured image fails
-Or the characters are incorrectly recognized
•	This can be overcome by using different modes of page segmentations which are in-built parameters provided by pytesseract.
•	Each segmentation mode is specific to a different kind of input format such as no text, one-line text, paragraph, page, table etc.
