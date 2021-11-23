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

LANGUAGE_RUS = 'rus'

SUCCESS = 0
FAILURE = 1

LEGACY = 0
LSTM = 1

PSM_TESSERACT = 3
PSM_OTHER = 6

TESSERACT_OSD = "tesseract_osd"
MY_OSD = "my_osd"

RECOGNIZED_DIR = 'recognized_text'
RECOGNIZED_DIR_AA = 'recognized_text_single'
ROTATED_IMAGES_DIR = 'rotated_images'

AA = False

log = Logger(logfile=f'{__file__}.log')

def psm_to_string(psm):
    if psm == PSM_TESSERACT:
        return TESSERACT_OSD
    if psm == PSM_OTHER:
        return MY_OSD

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
        - dict : metrics -> {"engine_type":{"wer":0, "cer":0}} !!!!!not relevant
    """
    try:
       
        start_time = time.perf_counter() # какйо-то значение секунд в данный момент

        if engine==LEGACY:
            engine_type = 'Legacy'
        elif engine==LSTM:
            engine_type = 'LSTM'

        output_dir = os.path.join(path_to_dataset, RECOGNIZED_DIR if not AA else RECOGNIZED_DIR_AA) #'dataset\\recognized_text'
        check_path(output_dir)
        output_dir_engine = os.path.join(output_dir, engine_type) #e.g. 'dataset\\recognized_text\\LSTM'
        check_path(output_dir_engine)
        path_to_rotated_images_dir = os.path.join(path_to_dataset, ROTATED_IMAGES_DIR) # 'dataset\\rotated_images'
        check_path(path_to_rotated_images_dir)

        path_to_text = data['text_path']
        with open(path_to_text, 'r', encoding='utf-8') as f:
            original_text = f.readlines()
            original_text = ' '.join(original_text)

        path = data['img_path']
        original_image = cv2.imread(path)

        # ПРЕДОБРАБОТКА МОЯ - MY_OSD

        img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        log.Info("Convert image from RGB to GRAY")
        #print('Image Dimensions :', img.shape, type(img))

        blur = cv2.GaussianBlur(img, (3, 3), sigmaX=1) # 
        #blur = cv2.medianBlur(img, 3) # 
        #blur = cv2.erode(blur, np.ones((5, 5), 'uint8'))
        # делать ли эрозию? 
        th1 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        ######
        mask = cv2.inRange(blur, 0, 150) #маска для инверсии
        #res = 255 - mask # инвертируем обратно
        coords = np.column_stack(np.where(mask > 0))
        #поворот изображения
        #th2 = cv2.bitwise_not(th1) # инвертирование
        #thresh = cv2.threshold(th2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        ######

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

        # структура ответа
        out_data = {
            engine_type : {
                #psm==1 - auto recog w/ OSD
                TESSERACT_OSD : {
                    # wer : 
                    # cer :
                },
                #psm==6 - recog block of text w/o OSD
                MY_OSD : {
                    # wer : 
                    # cer :
                }
            }
        }

        for psm_mode in [PSM_TESSERACT, PSM_OTHER]:
            # конфиг
            config = f"--oem {engine} --psm {psm_mode}"
            log.Info("Config(tesseract) for recognition: " + config)
            # распознаем текст по средствам тессеракта
            recognized = pytesseract.image_to_string(
                image  = (original_image if psm_mode==PSM_TESSERACT else rotated),
                lang   = LANGUAGE_RUS, 
                config = config
            )
            # сохранить распознанный текст
            filename = f"{data['id']}-{psm_to_string(psm_mode)}{TXT}"
            path_to_savefile = os.path.join(output_dir_engine, filename)
            log.Info("Writing recognized text to a directory")
            with open(path_to_savefile, 'w', encoding='utf-8') as f:
                f.write(recognized)
            # очищаем от лишних пробелов и одиночных знаков препинания
            orig_text = clean_string(original_text) 
            rec_text = clean_string(recognized)
            # рассчет метрик
            wer0 = mywer.wer2(orig_text, rec_text)
            cer0 = mywer.wer2(orig_text, rec_text, char_level=True)
            log.Info(f'Metrics: WER: {wer0} CER: {cer0}; lenghts: original = {len(orig_text)}, recognized = {len(rec_text)}')
            # обновляем out_data
            ret = { psm_to_string(psm_mode) : { 'wer' : wer0, 'cer' : cer0 } }
            out_data[engine_type].update(ret)

        end_time = time.perf_counter() - start_time
        log.Info(f"single element processing time: {end_time} sec")

        return out_data

    except FileNotFoundError as e:
        log.Error(f"error:{e}")
        sys.exit(1)

    except Exception as e:
        log.Error(f'{e.__class__}: {e}')
        sys.exit(1)

def conversion(sec):
   sec_value = sec % (24 * 3600)
   hour_value = sec_value // 3600
   sec_value %= 3600
   mins = sec_value // 60
   sec_value %= 60
   return f"{hour_value}h:{mins}m"

DATASET_JSON = 'dataset_info.json'
DATASET_JSON_AA = 'dataset_info_angle_analyze.json'
def main(engine=LSTM):

    # если используем старый движок
    if engine==LEGACY:
        TESSERACT_EXEC = "Tesseract-OCR\\Tesseract-OCR-5-legacy\\tesseract.exe"
    # если используем нейросети
    elif engine==LSTM:
        TESSERACT_EXEC = "Tesseract-OCR\\Tesseract-OCR-5-lstm-best\\tesseract.exe"
    else:
        raise

    pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXEC
    log.Info('pytesseract var. : ' + str(pytesseract.get_tesseract_version()))

    path_to_dataset = 'dataset'
    dataset_info_path = os.path.join(path_to_dataset, DATASET_JSON if not AA else DATASET_JSON_AA)

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
    main(engine=LSTM)
    del log
