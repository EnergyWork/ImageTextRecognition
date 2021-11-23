#!/usr/bin/python
# -*- coding: utf-8 -*-

import json
import os
import random
import sys
import textwrap
from time import sleep
from cv2 import rotate

import numpy as np
import skimage
import skimage.io
from PIL import Image, ImageDraw, ImageFont

from get_text_api import get_some_text
from logger.logger import Logger

JPG = '.jpg'
TXT = '.txt'
JSON = '.json'

FONTS_DIR = 'fonts'
IMAGES_DIR = 'images'
IMAGES_DIR_SINGLE = 'images_single'
ORIGINAL_TEXTS_DIR = 'original_texts'

SUCCESS = 0
FAILURE = 1

log = Logger(logfile=f'{__file__}.log')

def check_path(path, dir_name='DIR'):
    if not os.path.isdir(path):
        log.Warning(f'Каталог {dir_name} отсутствует, пробуем создать')
        try:
            os.mkdir(path)
        except OSError as e:
            log.Error(f'Не удалось создать каталог {dir_name}: {e}')
            sys.exit(FAILURE)

def get_text():
    error, text = get_some_text() #получаем текст
    if error:
        log.Error(text)
        sys.exit(1)
    return text

def get_wraped_text(text, line_width=60):
    return textwrap.wrap(text, width=line_width)

def create_dataset_element(
    path_to_dataset, 
    wraped_text, 
    save_name='unnamend', 
    rotate=True, 
    angle=999, # 999 значит, что не анализ угла
    with_noise=True
    ):
    """
    Функция для создания однгого элемента датасета
        -
        -

    return:
        -
    """

    log.Info(f'generate dataset element: {save_name}')

    angle_analysis = False if angle==999 else True

    try:
        #создаем стиль со шрифтом
        path_to_fonts = FONTS_DIR # рядом со скриптом, независимо от каталога датасета
        if not os.path.isdir(path_to_fonts):
            log.Warning('Каталог со шрифтами не обнаружен. Будет применём шрифт по умолчанию.')
            font = None
        else:
            fonts = os.listdir(path_to_fonts)
            fonts = list(filter(lambda x: x.endswith('.ttf'), fonts))
            selected_font = random.choice(fonts) if not angle_analysis else "Consolas.ttf"
            font = ImageFont.truetype(os.path.join(path_to_fonts, selected_font), size=18, encoding='utf-8') 
    except IOError as e:
        log.Warning('IOError:' + e)
        font = None
    except Exception as e:
        log.Warning('Unhandeled exception:' + e)
        font = None

    img = Image.new('RGB', (1000, 1000), color='white') #создаем изображение с белым фоном
    draw = ImageDraw.Draw(img) #объект для рисования
    
    margin = 50
    offset = 50 #начальный отступ слева и свкрху
    for line in wraped_text: #разделяем текст на строки
        draw.text((margin, offset), line, fill='black', font=font) #и добавляем на белом фоне текст
        offset += font.getsize(line)[1] + 5 # делая отступ по вертикали

    width, height = img.size
    max_len_line = sorted(wraped_text, key=(lambda x : len(x)), reverse=True)[0]
    max_width = font.getlength(max_len_line)
    img = img.crop((0, 0, max_width+margin+25, offset+50)) #обрезаем лишний белый фон

    # ПОВОРОТ ТЕКСТА 
    if rotate:
        if not angle_analysis: # рандомный угол как обычно, иначе угол из параметра
            angle = random.randint(-45, 45) # разброс угла поворота текста
        img = img.rotate(angle=angle, expand=True, fillcolor='white') # поворочиваем текст
        width, height = img.size 
        img = img.crop((15, 40, width-20,height-30)) #и снова обрезаем лишние зоны

    # ДОБАВЛЕНИЕ ШУМА
    img2 = np.array(img) #преобразуем иображение в массив ndarray из PIL.Image
    mode = 'none'
    if with_noise:
        noise_modes = ['gaussian', 'salt', 'speckle', 's&p', 'poisson']
        mode = random.choice(noise_modes) if not angle_analysis else 'speckle' # гвоздь
        img2 = skimage.util.random_noise(img2, mode=mode) #добавляем шум на изображение
    
    # СОХРАНЯЕМ
    p = IMAGES_DIR if not angle_analysis else IMAGES_DIR_SINGLE
    path_to_images = os.path.join(path_to_dataset, p)
    check_path(path_to_images, p)

    save_image_name = save_name + JPG
    path_to_image = os.path.join(path_to_images, save_image_name)
    try:
        skimage.io.imsave(path_to_image, (img2*255).astype(np.uint8)) #сохраняем изображение
    except Exception as e:
        log.Error(f'Не удалось сохранить изображение {save_image_name}; {e}')
        sys.exit(1)

    path_to_texts = os.path.join(path_to_dataset, ORIGINAL_TEXTS_DIR)
    check_path(path_to_texts, ORIGINAL_TEXTS_DIR)

    '''save_text_name = save_name + TXT
    path_to_text = os.path.join(path_to_texts, save_text_name)
    with open(path_to_text, 'w', encoding='utf-8') as f:
        tmp = [l+'\n' for l in wraped_text]
        f.writelines(tmp)'''
    
    data = {
        "img_path" : path_to_image,
        "text_path" : "",
        "font" : font.getname()[0],
        "noise" : mode,
        "metrics" : {},
    }
   
    return data
    

# synthetic dataset generation
COUNT_OF_DATASET_ELEMENTS = 50
COUNT_OF_ANGLES = 15
SLEEP_TIME = 0.15 # because fish-text.ru could ban, min: 0.1

def main_generator(angle_analyze=False):
    save_flag = False
    path_to_dataset = "dataset"
    check_path(path_to_dataset, path_to_dataset) #? KEKW
 
    elements_dict = []
    wraped_text = get_wraped_text(get_text()) # начальный текст, если для анализа угла
    for i in np.arange(0, COUNT_OF_DATASET_ELEMENTS if not angle_analyze else COUNT_OF_ANGLES):
        sleep(SLEEP_TIME)
        if not angle_analyze: 
            wraped_text = get_wraped_text(get_text()) # если надо разные тексты для каждого элемента
        element_dict = { "id" : str(i) }
        element_dict.update(
            create_dataset_element(
                path_to_dataset=path_to_dataset,
                wraped_text=wraped_text,
                save_name=str(i), 
                rotate=(False if not angle_analyze else True),
                angle=(i if angle_analyze else 999),
                with_noise=True,
            )
        )
        
        # сохраняем текст
        if not save_flag:
            save_text_name = (str(i) + TXT if not angle_analyze else 'single.txt')
            path_to_text = os.path.join(path_to_dataset, ORIGINAL_TEXTS_DIR, save_text_name)
            with open(path_to_text, 'w', encoding='utf-8') as f:
                tmp = [l+'\n' for l in wraped_text]
                f.writelines(tmp)
            save_flag = True if angle_analyze else False

        element_dict.update({"text_path" : path_to_text})
        # добавляем в массив словарей
        elements_dict.append(element_dict)
    
    try:
        DATASET_JSON = 'dataset_info.json' if not angle_analyze else 'dataset_info_angle_analyze.json'
        with open(os.path.join(path_to_dataset, DATASET_JSON), 'w', encoding="utf-8") as f:
            json.dump(elements_dict, f, ensure_ascii=False, indent=4)

    except OSError as e:
        log.Error(f'{e.__class__}: {e.__str__}')
    except TypeError as e:
        log.Error(f'{e.__class__}: {e.__str__}')
    except Exception as e:
        log.Error(f'{e.__class__}: {e.__str__}')


if __name__ == "__main__":
    main_generator(angle_analyze=False)
