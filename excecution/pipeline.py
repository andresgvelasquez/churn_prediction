from utils.functions import read_csv_files
from preprocessing.preprocessing import preprocessing_data
from preprocessing.preparing import preparing_data
from models.d00_dummy import dummytest
from models.m00_logistic_regresion import log_reg_model
from models.c00_catboost import catboost_model
from models.l00_lightgbm import lgbm_model
from models.x00_xgboost import xgboost_model

# Lectura de los archivos
df_contract, df_internet, df_personal, df_phone = read_csv_files('./files/datasets/input/')

# Preprocesar los datasets
df_telecom_clean = preprocessing_data(df_contract, df_internet, df_personal, df_phone)

# Preparar los datos
features_train_encoded, features_test_encoded, features_train_encoded_scaled, features_test_encoded_scaled, target_train, target_test = preparing_data(df_telecom_clean)

# Aplicar los modelos de machine learning

# Dummy model
dummytest(features_train_encoded_scaled, target_train, features_test_encoded_scaled, target_test)

# Logistic Regression model
log_reg_model(features_train_encoded_scaled, target_train, features_test_encoded_scaled, target_test)

# Catboost model
catboost_model(features_train_encoded_scaled, target_train, features_test_encoded_scaled, target_test)

# LightGBM model
lgbm_model(features_train_encoded_scaled, target_train, features_test_encoded_scaled, target_test)

# XGBoost model
xgboost_model(features_train_encoded_scaled, target_train, features_test_encoded_scaled, target_test)