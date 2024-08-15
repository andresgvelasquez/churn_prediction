from utils.functions import camelcase_to_snakecase, contract_cleaning, internet_cleaning, merge_datasets, personal_cleaning, phone_cleaning

def preprocessing_data(df_contract, df_internet, df_personal, df_phone):
    # Formatear el nombre de las columnas a snake_case para cada dataframe
    datasets = [df_contract, df_internet, df_personal, df_phone] 
    for df in datasets:
        camelcase_to_snakecase(df)

    # Limpiar cada dataset
    df_contract = contract_cleaning(df_contract)
    df_internet = internet_cleaning(df_internet)
    df_personal = personal_cleaning(df_personal)
    df_phone = phone_cleaning(df_phone)

    # Combinar los dataframes
    df_merged = merge_datasets(df_contract, df_internet, df_personal, df_phone)

    # Eliminar columnas innecesarias para el modelo
    df_merged.drop(['begin_year', 'begin_month', 'end_month', 'end_year', 'customer_id'], axis=1, inplace=True)

    return df_merged