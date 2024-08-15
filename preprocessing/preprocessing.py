from utils.functions import camelcase_to_snakecase, contract_preprocessing, internet_preprocessing, merge_datasets, personal_preprocessing, phone_preprocessing

def preprocessing_datasets(df_contract, df_internet, df_personal, df_phone):
    # Formatear el nombre de las columnas a snake_case para cada dataframe
    datasets = [df_contract, df_internet, df_personal, df_phone] 
    for df in datasets:
        camelcase_to_snakecase(df)

    # Limpiar cada dataset
    df_contract = contract_preprocessing(df_contract)
    df_internet = internet_preprocessing(df_internet)
    df_personal = personal_preprocessing(df_personal)
    df_phone = phone_preprocessing(df_phone)

    # Combinar los dataframes
    df_telecom_clean = merge_datasets(df_contract, df_internet, df_personal, df_phone)

    return df_telecom_clean