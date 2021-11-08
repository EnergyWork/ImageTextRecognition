#!/usr/bin/python
# -*- coding: utf-8 -*-

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
from logger.logger import Logger

import mywer

DATASET_JSON = 'dataset_info.json'
RECOGNIZED_DIR = 'recognized_text'
ROTATED_IMAGES_DIR = 'rotated_images'
TESSERACT_EXEC = "Tesseract-OCR\\Tesseract-OCR v5\\tesseract.exe"

log = Logger(logfile=f'{__file__}.log')
pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXEC

def clean_string(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.split()
    text = [word.lower() for word in text if word not in string.punctuation]
    return text

def recognize_function(path_to_dataset, json_file, detect_text_method=1, engine=0):
    """
    Recognize function

    Args:
        - path_to_dataset - path to dataset
        - json_file - name of json file
        - detect_text_method - 1 or 2
        - engine - 0, 1, 2
    
    Returns: 
        - None
    """
    try:
       
        path_to_file = os.path.join(path_to_dataset, JSONS_DIR, json_file) #'dataset3\\jsons\\filename.json'
        output_dir = os.path.join(path_to_dataset, RECOGNIZED_DIR) #'dataset3\\recognized_text'
        path_to_rotated_images_dir = os.path.join(path_to_dataset, ROTATED_IMAGES_DIR) # 'dataset3\\rotated_images'

        with open(path_to_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        path_to_text = data['text_path']
        with open(path_to_text, 'r', encoding='utf-8') as f:
            original_text = f.readlines()
            original_text = ' '.join(original_text)

        path = data['img_path']
        original_image = cv2.imread(path)
        img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        #print('Image Dimensions :', img.shape, type(img))

        blur = cv2.GaussianBlur(img, (3, 3), sigmaX=1)
        #blur = cv2.medianBlur(img, 3)
        ret1, th1 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        mask = cv2.inRange(blur, 0, 150) #маска для инверсии
        res = 255 - mask # инвертируем обратно
        coords = np.column_stack(np.where(mask > 0))

        #поворот изображения
        th2 = cv2.bitwise_not(th1)
        thresh = cv2.threshold(th2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        ##
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
        ##

        # draw_coords(points, img.shape)
        # draw_coords(coords, img.shape)

        # Angle varies between 0 to 90: https://namkeenman.wordpress.com/2015/12/18/open-cv-determine-angle-of-rotatedrect-minarearect/

        # Работает только для синтетического датасета.
        # coords - через маску для отсечения пикселей не проходящик через порог
        # points - через поиск контуров и отсечение контуров с маленькой площадью, в итоге будут контуры текста
        if detect_text_method == 1:
            rect = cv2.minAreaRect(points)
        elif detect_text_method == 2:
            rect = cv2.minAreaRect(coords)
        else:
            rect = cv2.minAreaRect(points)
        
        angle = rect[2]
        #print("[PRE-INFO] angle: {:.3f}".format(angle))
        (h, w) = cv2.boxPoints(rect).shape[:2] # box = np.int0(box)
        if angle < 45:
            h, w = w, h
        if w < h:
            angle = (90 - angle)
        else:
            angle = -angle
        #print("[INFO] angle: {:.3f}".format(angle))

        (h, w) = th1.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center=center, angle=(angle), scale=1.0)
        rotated = cv2.warpAffine(th1, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        filename = json_file.split('.')[0] + '.jpg'
        path_to_rot_img = os.path.join(path_to_rotated_images_dir, filename)
        cv2.imwrite(path_to_rot_img, rotated)

        # show images
        #cv2.imshow("Original", original_image)
        #cv2.imshow('Original with GBlur+threshhold', th1)
        #cv2.imshow("Rotated", rotated)

        #распознать текст
        #pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXEC
        #print(pytesseract.get_tesseract_version())
        #print(pytesseract.get_languages())
        config = f"--oem {engine} --psm 6"
        log.Info("current config: " + config)
        recognized  = pytesseract.image_to_string(rotated, lang='rus', config=config)

        filename = json_file.split('.')[0] + '.txt'
        path_to_savefile = os.path.join(output_dir, filename)
        with open(path_to_savefile, 'w', encoding='utf-8') as f:
            f.write(recognized)

        # считаем метрики и записываем в джейсон
        orig_text = clean_string(original_text) 
        rec_text = clean_string(recognized)
        wer0 = mywer.wer2(orig_text, rec_text)
        cer0 =  mywer.wer2(orig_text, rec_text, char_level=True)

        print(f'WER: {wer0} CER: {cer0}; lenghts: original = {len(orig_text)}, recognized = {len(rec_text)}')

        engine_type = 'baseline'

        data['metrics'][engine_type] = { 'wer' : wer0, 'cer' : cer0 }
        with open(path_to_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    except Exception as e:
        log.Error(e.__str__)

def main():
    log.Warning("#"*100)
    log.Info('pytesseract ver. ' + str(pytesseract.get_tesseract_version()))

    path_to_dataset = 'dataset'
    dataset_info_path = os.path.join(path_to_dataset, DATASET_JSON)
    log.Info(dataset_info_path)
    with open(dataset_info_path, 'r', encoding='utf-8') as f:
        dataset_info = json.load(f)
    for element in dataset_info:
        #recognize_function(path_to_dataset, json_file, detect_text_method=1)
        log.Info(element)

if __name__ == '__main__':
    main()
    del log
