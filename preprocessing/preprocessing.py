import pandas as pd
from utils.functions import camelcase_to_snakecase, contract_preprocessing

def preprocessing_datasets(df_contract, df_internet, df_personal, df_phone):
    # Formatear el nombre de las columnas a snake_case para cada dataframe
    datasets = [df_contract, df_internet, df_personal, df_phone] 
    for df in datasets:
        camelcase_to_snakecase(df)

    # Preprocesar cada dataset
    df_contract = contract_preprocessing(df_contract)

    return df_contract, df_internet, df_personal, df_phone