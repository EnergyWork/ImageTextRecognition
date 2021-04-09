import json
import os
import random
import textwrap
from time import sleep

import cv2
import numpy as np
import skimage
import skimage.io
from PIL import Image, ImageDraw, ImageFont

from get_text_api import get_some_text

FONTS_DIR = 'fonts'
IMAGES_DIR = 'images'
ORIGINAL_TEXTS_DIR = 'original_texts'
JSONS_DIR = 'jsons'

def check_path(path, dir_name):
    if not os.path.isdir(path):
        try:
            os.mkdir(path)
            return True
        except OSError as e:
            print(f'Не удалось создать каталог {dir_name}')
            return False

def create_dataset_element(path_to_dataset, save_name=None):
    error, text = get_some_text() #получаем текст
    if not error:
        img = Image.new('RGB', (1000, 1000), color='white') #создаем изображение с белым фоном
        try:
            #создаем стиль со шрифтом
            path_to_fonts = os.path.join(path_to_dataset, FONTS_DIR)
            if not os.path.isdir(path_to_fonts):
                print('Каталог со шрифтами не обнаружен. Будет применём шрифт по умолчанию.')
                font = None
            else:
                fonts = os.listdir(path_to_fonts)
                fonts = list(filter(lambda x: x.endswith('.ttf'), fonts))
                selected_font = random.choice(fonts);print(selected_font, end=' ')
                font = ImageFont.truetype(os.path.join(path_to_fonts, selected_font), size=22, encoding='unic') # 
        except IOError as e:
            print('IOError:', e)
            font = None
        except Exception as e:
            print(e)
            font = None

        draw = ImageDraw.Draw(img) #объект для рисования
        
        margin = offset = 50 #начальный отступ слева и свкрху
        line_width = 60 #количество символов в строке
        wraped_text = textwrap.wrap(text, width=line_width)
        for line in wraped_text: #разделяем текст на строки
            draw.text((margin, offset), line, fill='black', font=font) #и добавляем на белом фоне текст
            offset += font.getsize(line)[1] + 5 # делая отступ по вертикали

        width, height = img.size
        max_len_line = sorted(wraped_text, key=(lambda x : len(x)), reverse=True)[0]
        max_width = font.getlength(max_len_line)
        img = img.crop((0, 0, max_width + margin, offset + 50)) #обрезаем лишний белый фон

        angle = random.randint(-45, 45) #разброс угла поворота текста
        rotate_img = img.rotate(angle=angle, expand=True, fillcolor='white') #поворочиваем текст

        width, height = rotate_img.size 
        rotate_img = rotate_img.crop((15, 40, width-20,height-30)) #и снова обрезаем лишние зоны

        noise_modes = ['gaussian', 'localvar', 'poisson', 'salt', 'speckle', 's&p'] # поиграть с параметрами для разным методов, чтобы получать разные шумы
        mode = random.choice(noise_modes);print(mode)

        img2 = np.array(rotate_img) #преобразуем иображение в массив ndarray из PIL.Image
        gimg = skimage.util.random_noise(img2, mode=mode) #добавляем шум на изображение
        
        path_to_images = os.path.join(path_to_dataset, IMAGES_DIR)
        if not check_path(path_to_images, IMAGES_DIR):
            return

        save_name = str(save_name) if save_name is not None else 'NoneName'
        save_image_name = save_name + '.jpg'
        save_json_name = save_name + '.json'
        save_text_name = save_name + '.txt'

        path_to_image = os.path.join(path_to_images, save_image_name)
        try:
            skimage.io.imsave(path_to_image, (gimg*255).astype(np.uint8)) #сохраняем изображение
        except Exception:
            print('Не удалось сохранить изображение')
            return

        path_to_texts = os.path.join(path_to_dataset, ORIGINAL_TEXTS_DIR)
        if not check_path(path_to_texts, ORIGINAL_TEXTS_DIR):
            return

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
        
        path_to_jsons = os.path.join(path_to_dataset, JSONS_DIR)
        if not check_path(path_to_jsons, JSONS_DIR):
            return

        path_to_json = os.path.join(path_to_jsons, save_json_name)
        with open(path_to_json, 'w', encoding="utf-8") as f:
            json.dump(data, f)

    else:
        print(f"Error: {text}") 

if __name__ == "__main__":
    path_to_dataset = "dataset3"
    if not os.path.isdir(path_to_dataset):
        print('Неверно указан путь')
    else:
        for i in np.arange(0, 1):
            sleep(0.1)
            create_dataset_element(path_to_dataset=path_to_dataset, save_name=i)
