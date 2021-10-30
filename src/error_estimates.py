import os, json
import numpy as np


DATASET = 'dataset3'
JSONS_DIR = 'jsons'
JSONS_PATH = os.path.join(DATASET, JSONS_DIR)
WERs = []
CERs = []

def get_metrics(engine='baseline'):
    files = [json_file for json_file in os.listdir(JSONS_PATH) if json_file.endswith('.json')]
    for file in files:
        ofile = os.path.join(JSONS_PATH, file)
        try:
            with open(ofile, 'r', encoding='utf-8') as f:
                data = json.load(f)
                WERs.append(data['metrics'][engine]['wer'])
                CERs.append(data['metrics'][engine]['cer'])
        except KeyError as ke:
            print(f'{file}: нет такого ключа {str(ke)}')
        except Exception as e:
            print(f'{file}: {str(e)}')

def main():
    get_metrics()
    print(np.mean(WERs))
    print(np.mean(CERs))

if __name__ == '__main__':
    main()
