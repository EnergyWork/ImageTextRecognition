#!/usr/bin/python
# -*- coding: utf-8 -*-
#РЫБАТЕКСТ - fish-text.ru

import requests


#type - paragraph, number - 3, format - json : при этих параметрах вернется три абзатся в json-формате
URL = 'https://fish-text.ru/get'
params = {
    'type' : 'paragraph',
    'number' : 1,
    'format' : 'json'
}

fishtext_errors = {
    11 : {
        'http' : 200,
        'value' : 'Превышен допустимый объём запрашиваемого контента'
    },
    21 : {
        'http' : 403,
        'value' : 'IP заблокирован на 120 секунд из-за превышения лимита обращений'
    },
    22 : {
        'http' : 403,
        'value' : 'IP заблокирован навсегда'
    },
    31 : {
        'http' : 500,
        'value' : 'Неизвестная ошибка сервера'
    }
}

def get_some_text():
    """
    функция возвращает один абзац текста
        - return tuple:
            - error : true/false
            - text : текст ошибки либо абзац текста
    """
    error = False
    text = ''
    try:
        res = requests.get(URL, params=params)
        if res.status_code == 200:
            res_json = res.json()
            if res_json['status'] == 'success':
                text = res_json['text']
            else:
                text = f"Error: {res_json['errorCode']} - {fishtext_errors[res_json['errorCode']]['value']}"
                error = True
        else:
            text = f"Request error: {res.status_code}"
            error = True
    except Exception as e:
        text = str(e)
        error = True
    finally:
        return (error, text)

#test
if __name__ == '__main__':
    text = get_some_text()[1]
    with open('ttt.txt', 'w', encoding='utf-8') as f:
        f.write(text)