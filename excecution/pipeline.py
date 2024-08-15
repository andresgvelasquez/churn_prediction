from utils.functions import read_csv_files
from preprocessing.preprocessing import preprocessing_datasets

# Lectura de los archivos
df_contract, df_internet, df_personal, df_phone = read_csv_files('./files/datasets/input/')

# Preprocesar los datasets
df_telecom_clean = preprocessing_datasets(df_contract, df_internet, df_personal, df_phone)

print(df_telecom_clean.info())