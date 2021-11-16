import os, json
import numpy as np
from logger.logger import Logger

log = Logger(logfile=f'{__file__}.log')

DATASET = 'dataset'
DATASET_JSON = 'dataset_info.json'

WERs = []
CERs = []

def get_metrics(engine='Legacy'):
    # читаем файл с данными по распознованию
    with open(os.path.join(DATASET, DATASET_JSON), "r", encoding="utf-8") as f:
        file_data = json.load(f)
    
    try:
        for data in file_data:
            WERs.append(data['metrics'][engine]['wer'])
            CERs.append(data['metrics'][engine]['cer'])
    except KeyError as ke:
        log.Info(f'unknown key : {str(ke)}')

def main():
    # engine_type = 'Legacy'
    # engine_type = 'LSTM'
    for engine in ['Legacy', 'LSTM']:
        get_metrics(engine)
        log.Info(f"{engine}: mean WER: {np.mean(WERs)}")
        log.Info(f"{engine}: mean CER: {np.mean(CERs)}")

if __name__ == '__main__':
    main()
