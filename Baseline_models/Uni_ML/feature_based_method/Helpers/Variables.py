import torch


device = torch.device(f'cuda:3' if torch.cuda.is_available() else 'cpu')



RAW_DIR = '/opt/workspace/Seohyeon/Journal/DATA/matfile'

DATASET_DIR = '/opt/workspace/Seohyeon/Journal/DATA/preprocessed'

WD = f'/opt/workspace/Seohyeon/Journal/Other_methods/Baseline/feature_based_method/' # 'or' WD = os.getcwd()


METRICS = ['loss', 'acc', 'bacc', 'f1', 'preci', 'recall']


COL_NAMES = [
    'Fp1', 'Fp2', 'AF3', 'AF4', 'F7', 'F8', 'F3', 'Fz', 'F4', 'FC5', 
    'FC6', 'T7', 'T8', 'C3', 'C4', 'CP5', 'CP6', 'P7', 'P8', 'P3', 
    'Pz', 'P4', 'PO7', 'PO8', 'PO3', 'PO4', 'O1', 'O2', 
    'ECG', 'Resp', 'PPG', 'GSR'
]


# 파일명
FILENAME_MODEL = f'MS__model.pt'
FILENAME_HIST = f'MS__history.csv'
FILENAME_HISTSUM = f'MS__history_summary.csv'
FILENAME_RES = f'MS__result.csv'

FILENAME_TOTALSUM = f'MS__total_summary.csv'
FILENAME_TOTALRES = f'MS__total_result.csv' 
FILENAME_FOLDSUM = f'MS__fold_summary.csv'


FILENAME_FOLDRES = f'MS__fold_result.csv'
FILENAME_FOLDSUM = f'MS__fold_summary.csv'
