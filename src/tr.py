#!/usr/bin/python
# -*- coding: utf-8 -*-

import json
import os
import re
import string
import sys
import time

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

JPG = '.jpg'
TXT = '.txt'
JSON = '.json'

SUCCESS = 0
FAILURE = 1

LEGACY = 0
LSTM = 1

DATASET_JSON = 'dataset_info.json'
RECOGNIZED_DIR = 'recognized_text'
ROTATED_IMAGES_DIR = 'rotated_images'

log = Logger(logfile=f'{__file__}.log')

def check_path(path):
    if not os.path.isdir(path):
        log.Warning(f'The {path} directory is missing, we are trying to create')
        try:
            os.mkdir(path)
        except OSError as e:
            log.Error(f'Failed to create directory {path}: {e}')
            sys.exit(FAILURE)

def clean_string(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.split()
    text = [word.lower() for word in text if word not in string.punctuation]
    return text

def recognize_function(path_to_dataset, data, detect_text_method=1, engine=0):
    """
    Recognize function

    Args:
        - path_to_dataset : path to dataset
        - data : name of json file
        - detect_text_method : 1 or 2
        - engine : 0, 1, 2
    
    Returns: 
        - dict : metrics -> {"engine_type":{"wer":0, "cer":0}}
    """
    try:
       
        start_time = time.perf_counter()


        output_dir = os.path.join(path_to_dataset, RECOGNIZED_DIR) #'dataset\\recognized_text'
        check_path(output_dir)
        path_to_rotated_images_dir = os.path.join(path_to_dataset, ROTATED_IMAGES_DIR) # 'dataset\\rotated_images'
        check_path(path_to_rotated_images_dir)

        path_to_text = data['text_path']
        with open(path_to_text, 'r', encoding='utf-8') as f:
            original_text = f.readlines()
            original_text = ' '.join(original_text)

        path = data['img_path']
        original_image = cv2.imread(path)

        img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        log.Info("Convert image from RGB to GRAY")
        #print('Image Dimensions :', img.shape, type(img))

        blur = cv2.GaussianBlur(img, (3, 3), sigmaX=1) # 
        #blur = cv2.medianBlur(img, 3) # 
        # делать ли эрозию? 
        ret1, th1 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        mask = cv2.inRange(blur, 0, 150) #маска для инверсии
        res = 255 - mask # инвертируем обратно
        coords = np.column_stack(np.where(mask > 0))

        #поворот изображения
        th2 = cv2.bitwise_not(th1) # инвертирование
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
            log.Warning(f"Unknown text detection method, the default method will be applied")
            rect = cv2.minAreaRect(points)
        
        angle = rect[2]
        log.Info("Text rotation angle: {:.3f}".format(angle))
        (h, w) = cv2.boxPoints(rect).shape[:2] # box = np.int0(box)
        if angle < 45:
            h, w = w, h
        if w < h:
            angle = (90 - angle)
        else:
            angle = -angle
        log.Info("The angle at which the text was rotated: {:.3f}".format(angle))

        (h, w) = th1.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center=center, angle=(angle), scale=1.0)
        rotated = cv2.warpAffine(th1, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        filename = data['id'] + JPG
        path_to_rot_img = os.path.join(path_to_rotated_images_dir, filename)
        log.Info("Writing an image with aligned text to a directory")
        cv2.imwrite(path_to_rot_img, rotated) # запись изображения в каталог

        # show images
        #cv2.imshow("Original", original_image)
        #cv2.imshow('Original with GBlur+threshhold', th1)
        #cv2.imshow("Rotated", rotated)

        #РАСПОЗВНОВАНИЕ

        # 0 = Original Tesseract only.
        # 1 = Neural nets LSTM only.

        if engine==LEGACY:
            engine_type = 'Legacy'
        elif engine==LSTM:
            engine_type = 'LSTM'

        config = f"--oem {engine} --psm 6" # psm==6 - block of text
        log.Info("Config(tesseract) for recognition: " + config)
        recognized = pytesseract.image_to_string(rotated, lang='rus', config=config)

        filename = f"{data['id']}-{engine_type}{TXT}"
        path_to_savefile = os.path.join(output_dir, filename)
        log.Info("Writing recognized text to a directory")
        with open(path_to_savefile, 'w', encoding='utf-8') as f:
            f.write(recognized)

        # считаем метрики и записываем в джейсон
        orig_text = clean_string(original_text) 
        rec_text = clean_string(recognized)

        wer0 = mywer.wer2(orig_text, rec_text)
        cer0 =  mywer.wer2(orig_text, rec_text, char_level=True)

        log.Info(f'Metrics: WER: {wer0} CER: {cer0}; lenghts: original = {len(orig_text)}, recognized = {len(rec_text)}')

        ret = { engine_type : { 'wer' : wer0, 'cer' : cer0 } }
        log.Info(f'return: {ret}')

        end_time = time.perf_counter() - start_time

        log.Info(f"single element processing time: {end_time} sec")

        return ret


    #except OSError as e: 
    #    log.Error(f'{e.__class__}: {e.__str__}')
    except Exception as e:
        log.Error(f'{e.__class__}: {e.__str__}')
        sys.exit(1)

def conversion(sec):
   sec_value = sec % (24 * 3600)
   hour_value = sec_value // 3600
   sec_value %= 3600
   mins = sec_value // 60
   sec_value %= 60
   return f"{hour_value}h:{mins}m"

def main(engine=LSTM):

    # если используем старый движок
    if engine==LEGACY:
        TESSERACT_EXEC = "Tesseract-OCR\\Tesseract-OCR-5-legacy\\tesseract.exe"
    # если используем лстам
    elif engine==LSTM:
        TESSERACT_EXEC = "Tesseract-OCR\\Tesseract-OCR-5-lstm-best\\tesseract.exe"
    else:
        raise

    pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXEC
    log.Info('pytesseract var. : ' + str(pytesseract.get_tesseract_version()))

    path_to_dataset = 'dataset'
    dataset_info_path = os.path.join(path_to_dataset, DATASET_JSON)

    log.Info(dataset_info_path)

    with open(dataset_info_path, 'r', encoding='utf-8') as f:
        dataset_info = json.load(f)

    st = time.perf_counter()

    for element in dataset_info:
        log.Info(f'Current element: {element}')
        metrics = recognize_function(path_to_dataset, element, detect_text_method=1, engine=engine)
        element["metrics"].update(metrics)

    log.Info(f"Total time: {conversion(time.perf_counter() - st)}")

    with open(dataset_info_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    main(engine=LEGACY)
    del log
