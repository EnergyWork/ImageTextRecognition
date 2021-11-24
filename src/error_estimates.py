import os, json, sys
import numpy as np
from logger.logger import Logger

log = Logger(logfile=f'{__file__}.log')

DATASET = 'dataset'
DATASET_JSON = 'dataset_info.json'
DATASET_JSON_AA = 'dataset_info_angle_analyze.json'

AA = True

WERs = []
CERs = []

def get_metrics(engine='Legacy', osd='tesseract_osd'):
    # читаем файл с данными по распознованию
    with open(os.path.join(DATASET, DATASET_JSON if not AA else DATASET_JSON_AA), "r", encoding="utf-8") as f:
        file_data = json.load(f)
    del WERs[:]
    del CERs[:]
    try:
        for data in file_data:
            WERs.append(data['metrics'][engine][osd]['wer'])
            CERs.append(data['metrics'][engine][osd]['cer'])
    except KeyError as ke:
        log.Info(f'unknown key : {str(ke)}')
        sys.exit(1)

def main():
    # engine_type = 'Legacy'
    # engine_type = 'LSTM'
    for engine in ['Legacy', 'LSTM']:
        for osd in ['tesseract_osd', 'my_osd']:
            get_metrics(engine, osd)
            log.Info(f"{engine}, {osd}: mean WER: {np.mean(WERs)}")
            log.Info(f"{engine}, {osd}: mean CER: {np.mean(CERs)}")


def aa_analysis():
    for engine in ['Legacy', 'LSTM']:
        get_metrics(engine, 'tesseract_osd')

        log.Info(f"{engine}: WER")
        for e in WERs:
            log.Info(f"{e:.4f}")

        log.Info(f"{engine}: CER")
        for e in CERs:
            log.Info(f"{e:.4f}")

if __name__ == '__main__':
    #main()
    aa_analysis()
