from utils.functions import read_csv_files
from preprocessing.preprocessing import preprocessing_data
from preprocessing.preparing import preparing_data

# Lectura de los archivos
df_contract, df_internet, df_personal, df_phone = read_csv_files('./files/datasets/input/')

# Preprocesar los datasets
df_telecom_clean = preprocessing_data(df_contract, df_internet, df_personal, df_phone)

# Preparar los datos
features_train_encoded, features_test_encoded, features_train_encoded_scaled, features_test_encoded_scaled = preparing_data(df_telecom_clean)

print(features_train_encoded.info())
print(features_train_encoded)
print(features_test_encoded.info())
print(features_test_encoded)
print(features_train_encoded_scaled.info())
print(features_train_encoded_scaled)
print(features_test_encoded_scaled.info())
print(features_test_encoded_scaled)