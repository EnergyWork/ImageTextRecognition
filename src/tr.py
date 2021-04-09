import os

import cv2
import fastwer
import matplotlib.pyplot as plt
import numpy as np
import pytesseract
from PIL import Image, ImageDraw

from pywer import wer


# def draw_coords(coords, img_shape, oldimg=None):
#     if oldimg is None:
#         img = Image.new('RGB', (img_shape[0], img_shape[1]), color='white')
#     else:
#         img = Image.fromarray(oldimg)
#     draw = ImageDraw.Draw(img)
#     ncoords = [(c[0], c[1]) for c in coords]
#     draw.point(ncoords, fill ='red')
#     ##############
#     rect = cv2.minAreaRect(coords) # пытаемся вписать прямоугольник
#     #draw.rectangle((rect[0], rect[1]), outline='black', width=2)
#     #draw.point(rect[1], fill ='black')
#     print(f'[DRAW_COORDS INFO] {rect[2]}')
#     box = cv2.boxPoints(rect) # поиск четырех вершин прямоугольника
#     box = np.int0(box) # округление координат
#     img = np.array(img)
#     cv2.drawContours(img, [box], 0, (0,0,255), 2) # рисуем прямоугольник
#     cv2.imshow('saaffa', img)
#     cv2.waitKey()
#     ##############
#     #img.show()


path = "dataset3\\0000.jpg" # fix path
original_image = cv2.imread(path)
img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
print('Image Dimensions :', img.shape, type(img))

blur = cv2.GaussianBlur(img, (3, 3), sigmaX=1)
ret1, th1 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

mask = cv2.inRange(blur, 0, 150) #маска для инверсии
res = 255 - mask # инвертируем обратно

#поворот изображения

th2 = cv2.bitwise_not(th1)
thresh = cv2.threshold(th2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

coords = np.column_stack(np.where(mask > 0))

contours0, hierarchy = cv2.findContours(image=th1.copy(), mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)
areas = [cv2.contourArea(cnt) for cnt in contours0]
indx = 0
contours0 = np.array(contours0)
for cnt in contours0:
    if cv2.contourArea(cnt) < 10 or cv2.contourArea(cnt) >= np.max(areas):
        contours0 = np.delete(contours0, indx)
    else:
        indx += 1
areas = [cv2.contourArea(cnt) for cnt in contours0]

points = []
for cnt in contours0:
    for p in cnt:
        points.append([p[0][1], p[0][0]])
points = np.array(points)

# draw_coords(points, img.shape)
# draw_coords(coords, img.shape)

# Angle varies between 0 to 90: https://namkeenman.wordpress.com/2015/12/18/open-cv-determine-angle-of-rotatedrect-minarearect/

# Работает только для синтетического датасета.
# coords - через маску для отсечения пикселей не проходящик через порог
# points - через поиск контуров и отсечение контуров с маленькой площадью, в итоге будут контуры текста
rect = cv2.minAreaRect(points) # coords
angle = rect[2] # 
print("[PRE-INFO] angle: {:.3f}".format(angle))
(h, w) = cv2.boxPoints(rect).shape[:2] # box = np.int0(box)
if angle < 45:
    h, w = w, h
if w < h:
    angle = (90 - angle)
else:
    angle = -angle
print("[INFO] angle: {:.3f}".format(angle))

(h, w) = th1.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center=center, angle=(angle), scale=1.0)
rotated = cv2.warpAffine(th1, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE) #

# show images
cv2.imshow("Original", original_image)
cv2.imshow('Original with GBlur+threshhold', th1)
cv2.imshow("Rotated", rotated)

#попытки распознать текст
pytesseract.pytesseract.tesseract_cmd = "Tesseract-OCR\\Tesseract-OCR v5\\tesseract.exe"
print(pytesseract.get_tesseract_version())
print(pytesseract.get_languages())
config = "--oem 0 --psm 6"
data = pytesseract.image_to_string(rotated, lang='rus', config=config)
with open('output\\rec.txt', 'w+', encoding='utf-8') as f:
    f.write(data)

cv2.waitKey()

#############################TESSARACT PARAMS INFO################################################

# --oem N
# Specify OCR Engine mode. The options for N are:
# 0 = Original Tesseract only.
# 1 = Neural nets LSTM only.
# 2 = Tesseract + LSTM.
# 3 = Default, based on what is available.

# --psm N
# Set Tesseract to only run a subset of layout analysis and assume a certain form of image. The options for N are:
# 0 = Orientation and script detection (OSD) only.
# 1 = Automatic page segmentation with OSD.
# 2 = Automatic page segmentation, but no OSD, or OCR.
# 3 = Fully automatic page segmentation, but no OSD. (Default)
# 4 = Assume a single column of text of variable sizes.
# 5 = Assume a single uniform block of vertically aligned text.
# 6 = Assume a single uniform block of text.
# 7 = Treat the image as a single text line.
# 8 = Treat the image as a single word.
# 9 = Treat the image as a single word in a circle.
# 10 = Treat the image as a single character.
