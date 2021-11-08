#!/usr/bin/python
# -*- coding: utf-8 -*-

import json
import os
import random
import sys
import textwrap
from time import sleep

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
ORIGINAL_TEXTS_DIR = 'original_texts'
DATASET_JSON = 'dataset_info.json'

COUNT_OF_DATASET_ELEMENTS = 2
SLEEP_TIME = 0.1 # because fish-text.ru could ban, min: 0.1

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

def create_dataset_element(path_to_dataset, save_name='unnamend'):
    """
    функция для создания однгого элемента датасета
        - path_to_dataset : путь до каталога, в который складывайть данные
        - save_name : имя для данных элемента датасета
    """

    log.Info(f'generate dataset element: {save_name}')

    error, text = get_some_text() #получаем текст
    if error:
        log.Error(text)

    img = Image.new('RGB', (1000, 1000), color='white') #создаем изображение с белым фоном
    try:
        #создаем стиль со шрифтом
        path_to_fonts = FONTS_DIR # рядом со скриптом, независимо от каталога датасета
        if not os.path.isdir(path_to_fonts):
            log.Warning('Каталог со шрифтами не обнаружен. Будет применём шрифт по умолчанию.')
            font = None
        else:
            fonts = os.listdir(path_to_fonts)
            fonts = list(filter(lambda x: x.endswith('.ttf'), fonts))
            selected_font = random.choice(fonts)
            font = ImageFont.truetype(os.path.join(path_to_fonts, selected_font), size=22, encoding='utf-8') 
    except IOError as e:
        log.Warning('IOError:' + e)
        font = None
    except Exception as e:
        log.Warning('Unhandeled exception:' + e)
        font = None

    draw = ImageDraw.Draw(img) #объект для рисования
    
    margin = offset = 50 #начальный отступ слева и свкрху
    line_width = 60 #количество символов в строке
    wraped_text = textwrap.wrap(text, width=line_width)
    for line in wraped_text: #разделяем текст на строки
        draw.text((margin, offset), line, fill='black', font=font) #и добавляем на белом фоне текст
        offset += font.getsize(line)[1] + 5 # делая отступ по вертикали

    width, height = img.size
    max_len_line = sorted(wraped_text, key=(lambda x : len(x)), reverse=True)[0] # длина самой длинной строки # ? чо???? хахаха что за бред??? просто np.max(wraped_text)????
    max_width = font.getlength(max_len_line)
    img = img.crop((0, 0, max_width + margin, offset + 50)) #обрезаем лишний белый фон

    angle = random.randint(-45, 45) #разброс угла поворота текста
    rotate_img = img.rotate(angle=angle, expand=True, fillcolor='white') #поворочиваем текст

    width, height = rotate_img.size 
    rotate_img = rotate_img.crop((15, 40, width-20,height-30)) #и снова обрезаем лишние зоны

    noise_modes = ['gaussian', 'localvar', 'poisson', 'salt', 'speckle', 's&p'] # поиграть с параметрами для разным методов, чтобы получать разные шумы
    mode = random.choice(noise_modes)

    img2 = np.array(rotate_img) #преобразуем иображение в массив ndarray из PIL.Image
    gimg = skimage.util.random_noise(img2, mode=mode) #добавляем шум на изображение
    
    path_to_images = os.path.join(path_to_dataset, IMAGES_DIR)
    check_path(path_to_images, IMAGES_DIR)

    save_image_name = save_name + JPG
    path_to_image = os.path.join(path_to_images, save_image_name)
    try:
        skimage.io.imsave(path_to_image, (gimg*255).astype(np.uint8)) #сохраняем изображение
    except Exception:
        log.Error(f'Не удалось сохранить изображение {save_image_name}')
        sys.exit(1)

    path_to_texts = os.path.join(path_to_dataset, ORIGINAL_TEXTS_DIR)
    check_path(path_to_texts, ORIGINAL_TEXTS_DIR)

    save_text_name = save_name + TXT
    path_to_text = os.path.join(path_to_texts, save_text_name)
    with open(path_to_text, 'w', encoding='utf-8') as f:
        tmp = [l+'\n' for l in wraped_text]
        f.writelines(tmp)
    
    data = {
        "img_path" : path_to_image,
        "text_path" : path_to_text,
        "font" : font.getname()[0],
        "noise" : mode,
    }

    '''path_to_jsons = os.path.join(path_to_dataset, JSONS_DIR)
    if not check_path(path_to_jsons, JSONS_DIR):
        return

    save_json_name = save_name + '.json';
    path_to_json = os.path.join(path_to_jsons, save_json_name)
    with open(path_to_json, 'w', encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)'''
    
    return data

# synthetic dataset generation
if __name__ == "__main__":
    log.Warning("#"*100)

    path_to_dataset = "dataset"
    check_path(path_to_dataset, path_to_dataset) #? KEKW
 
    elements_dict = []
    for i in np.arange(0, COUNT_OF_DATASET_ELEMENTS):
        sleep(SLEEP_TIME)
        element_dict = create_dataset_element(path_to_dataset=path_to_dataset, save_name=str(i))
        elements_dict.append({ str(i) : element_dict})
    
    try:
        
        with open(os.path.join(path_to_dataset, DATASET_JSON), 'w', encoding="utf-8") as f:
            json.dump(elements_dict, f, ensure_ascii=False, indent=4)

    except OSError as e:
        log.Error(f'{e.__class__}: {e.__str__}')
    except TypeError as e:
        log.Error(f'{e.__class__}: {e.__str__}')
    except Exception as e:
        log.Error(f'{e.__class__}: {e.__str__}')
