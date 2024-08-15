import pandas as pd

def read_csv_files(files_path:str, contract_name:str='contract.csv', internet_name:str='internet.csv', personal_name:str='personal.csv', phone_name:str='phone.csv'):
    '''
    Esta función permite leer los 4 archivos CSV en el siguiente orden: contrat, internet, personal y phone.
    files_path: Es el string de la carpeta que contienen los csv,
    contract_name: nombre del archivo del contrato por ejemplo 'contract.csv',
    internet_name: nombre del archivo de internet por ejemplo 'internet.csv',
    personal_name: nombre del archivo del personal por ejemplo 'personal.csv',
    phone_name: nombre del archivo de telefonia por ejemplo 'phone.csv'.

    '''

    df_contract = pd.read_csv(files_path+contract_name)
    df_internet = pd.read_csv(files_path+internet_name)
    df_personal = pd.read_csv(files_path+personal_name)
    df_phone = pd.read_csv(files_path+phone_name)
    return df_contract, df_internet, df_personal, df_phone