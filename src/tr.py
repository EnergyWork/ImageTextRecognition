import json
import os
import re
import string

import cv2
import fastwer
import jiwer
import Levenshtein
import matplotlib.pyplot as plt
import numpy as np
import pytesseract
from PIL import Image, ImageDraw

from mywer import wer


def split(word):
    return [char for char in word]

def clean_string(text, char_sentence=False):
    if char_sentence:
        text = split(re.sub(r'\s+', '', text))
    else:
        text = text.split()
    text = [word.lower() for word in text if word not in string.punctuation]
    return text

def wer2(ref, hyp):
    pass

def recognize_function(path_to_dataset):
    path_to_file = 'dataset3\\jsons\\0.json'
    output_dir = 'dataset3\\recognized_text'

    with open(path_to_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    path_to_text = data['text_path']
    with open(path_to_text, 'r', encoding='utf-8') as f:
        original_text = f.readlines()
        original_text = ' '.join(original_text)

    path = data['img_path']
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
    cv2.imwrite('dataset3\\roteted_images\\0.jpg', rotated)

    # show images
    #cv2.imshow("Original", original_image)
    #cv2.imshow('Original with GBlur+threshhold', th1)
    #cv2.imshow("Rotated", rotated)

    #распознать текст
    pytesseract.pytesseract.tesseract_cmd = "Tesseract-OCR\\Tesseract-OCR v5\\tesseract.exe"
    print(pytesseract.get_tesseract_version())
    print(pytesseract.get_languages())
    config = "--oem 0 --psm 6"
    recognized  = pytesseract.image_to_string(rotated, lang='rus', config=config)

    filename = '0.txt'
    path_to_savefile = os.path.join(output_dir, filename)
    with open(path_to_savefile, 'w', encoding='utf-8') as f:
        f.write(recognized )

    w_orig_text = clean_string(original_text) # original_text.lower().split()
    w_r_text = clean_string(recognized) # recognized.lower().split()
    wer1 = wer(w_orig_text, w_r_text) / max(len(w_orig_text), len(w_r_text))
    print('WER:', wer1, fastwer.score(w_r_text, w_orig_text), jiwer.wer(w_orig_text, w_r_text))

    c_orig_text = clean_string(original_text, char_sentence=True) # split(''.join(w_orig_text).lower())
    c_r_text = clean_string(recognized, char_sentence=True) # split(''.join(w_r_text).lower())
    cer1 = wer(c_orig_text, c_r_text) / max(len(c_orig_text), len(c_r_text))
    print('CER:', cer1, fastwer.score(c_r_text, c_orig_text, char_level=True), jiwer.wer(c_orig_text, c_r_text))

    data['metrics'] = {
        'wer' : int(wer1),
        'cer' : int(cer1)
    }
    with open(path_to_file, 'w', encoding='utf-8') as f:
        json.dump(data, f)

def main():
    path_to_dataset = 'dataset3'
    recognize_function(path_to_dataset)

def test_wer():
    h = clean_string('Mathworks connection programs', char_sentence=True);print(h)
    r = clean_string('MathWorks Connections Program in coal', char_sentence=True);print(r)
    print(len('Mathworks connection programs'), len('MathWorks Connections Program in coal'))
    print(len(h), len(r))
    import timeit

    res = timeit.timeit('wer(r, h) / len(r)', setup='from mywer import wer', number=1000, globals = locals()) #x1
    print(wer(r, h) / len(r), res)

    res = timeit.timeit('jiwer.wer(r, h)', setup='import jiwer', number=1000, globals = locals()) #x4
    print(jiwer.wer(r, h), res)

    res = timeit.timeit('fastwer.score(h, r)', setup='import fastwer', number=1000, globals = locals()) #x28
    print(fastwer.score(h, r, char_level=True), res)

if __name__ == '__main__':
    # main()
    test_wer()
