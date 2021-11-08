import json
import os

def Jsonload_test():
    with open("./src/test/test", 'r', encoding='utf-8') as f:
        data = json.load(f)

if __name__ == '__main__':
    print(os.getcwd())
    Jsonload_test()