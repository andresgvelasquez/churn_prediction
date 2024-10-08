import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_curve, roc_auc_score, average_precision_score, precision_recall_curve

def bool_user_active(user_info):
    ''' Verifica si la persona sigue activa.
    Si tiene NaT, significa que la persona esta activa y se pondra un True.
    En caso contrario se agrega un False.'''
    return pd.isna(user_info['end_date'])

def calculate_end_date_by_year(user_info, last_year=2020, add_n_years=None, begin_date='begin_date'):
    ''' Calcula la fecha de terminacion de contrato de 1 o mas años.'''
    # Contrato por años
    # Para los que empezaron en el año actual
    if (user_info[begin_date].year == last_year) & (add_n_years):
        return user_info[begin_date] + pd.DateOffset(years=add_n_years) # Agrega años
    # Para los que empezaron el año anterior
    else:
        return pd.to_datetime(f'{last_year-1}-{user_info["begin_date"].month}-01') + pd.DateOffset(years=add_n_years)
    
def calculate_active_days(user_info, begin_date='begin_date',end_point='2020-01-01'):
    ''' Calcula los días en las que un usuario estuvo activo, limitado por el último dia registrado
    de cancelación/terminación de contrato. '''
    end_active_day = user_info['end_date_with_contract']
    # Limitar la fecha al último día registrado
    if end_active_day > pd.to_datetime(end_point):
        end_active_day = pd.to_datetime(end_point)
    return (end_active_day - user_info[begin_date]).days

def camelcase_to_snakecase(df):
    ''' Convierte las columnas de un dataframe de formato CamelCase
    a snake_case y minúsculas.
    in: dataframe con columnas CamelCase
    out: dataframe con columnas snake_case '''
    columns = df.columns
    new_col_names = []
    for col in columns:
        pattern = r'([a-z])([A-Z])' # Encuentra minúscula seguida de una mayúscula
        replace = r'\1_\2'          # Reemplaza por minúscula_mayúscula
        snke_case_name = re.sub(pattern=pattern, repl=replace, string=col)
        new_col_names.append(snke_case_name.lower())
    df.columns = new_col_names

def dcolumn_to_bool(df, columns):
    ''' Convierte el tipo de dato de las columnas con 2 categorias a bool. 
    columns: Columnas a las cuales se les aplicara la transformación. '''
    df = pd.get_dummies(df, columns=columns, drop_first=True)
    for col in df.columns:
        pattern = r'_([A-Z].*)'     # Encuentra el patron que agrega get_dummies
        replace = ''                # El reemplazo es una cadena vacia
        clean_name = re.sub(pattern=pattern, repl=replace, string=col)
        df = df.rename(columns={col:clean_name})
    return df 

def contract_cleaning(df_contract, total_charges='total_charges',
                      end_date='end_date', begin_date='begin_date',
                      is_active='is_active', paperless_billing='paperless_billing',
                      end_date_with_contract='end_date_with_contract',
                      active_days='active_days'):
    
    # Reemplazar las cadenas vacías de total_charges por 0 y convertir la columna a tipo float
    df_contract[total_charges] = df_contract[total_charges].replace(' ', '0').astype(float)

    # Convertir las columna end_date/begin_date a datetime
    df_contract[end_date] = pd.to_datetime(df_contract[end_date], errors='coerce')
    df_contract[begin_date] = pd.to_datetime(df_contract[begin_date], errors='coerce')


    # Separar la columna begin_date de df_contract en año, mes
    df_contract = split_dates(df_contract, begin_date, 'begin')

    # Separar la columna end_date de df_contract en año, mes
    df_contract = split_dates(df_contract, end_date, 'end')

    # Crear una columna que diga si la persona esta activa o no, a partir de end_date
    df_contract[is_active] = df_contract.apply(bool_user_active, axis=1)

    # Convertir la columnas paperless_billing a tipo bool
    df_contract = dcolumn_to_bool(df_contract, [paperless_billing])

    # Reemplazar los valores ausentes de end_date por el vencimiento del contrato
    df_contract[end_date_with_contract] = df_contract.apply(real_end_date, axis=1)

    # Separar la columna end_date_with_contract de df_contract en año, mes
    # Se hace de nuevo porque antes era para el análisis y ahora necesitamos eliminar los valores ausentes para el modelo.
    df_contract = split_dates(df_contract, 'end_date_with_contract', 'end')

    # Crear una columna con los días activo en contract
    df_contract[active_days] = df_contract.apply(calculate_active_days, axis=1)

    # Eliminar columnas innecesarias: being_date, end_date, end_date_with_contract
    df_contract.drop([begin_date, end_date, end_date_with_contract], axis=1, inplace=True)
    return df_contract

def personal_cleaning(df_personal):
        
    # Cambiar las columnas gender, partner y dependents por bool. 
    df_personal = dcolumn_to_bool(df_personal, ['gender', 'partner', 'dependents'])
    df_personal = df_personal.rename(columns={'gender':'gender_male'}) 
    return df_personal

def internet_cleaning(df_internet):

    # Cambiar las columnas al tipo bool
    df_internet = dcolumn_to_bool(df_internet, ['internet_service', 'online_security', 'online_backup', 'device_protection',
                                            'tech_support', 'streaming_tv', 'streaming_movies'])
    df_internet = df_internet.rename(columns={'internet_service':'internet_fiber_optic'})

    return df_internet

def phone_cleaning(df_phone):
    # Cambiar la columna multiple_lines a tipo bool. 
    df_phone = dcolumn_to_bool(df_phone, ['multiple_lines'])
    return df_phone

def merge_datasets(df_contract, df_internet, df_personal, df_phone, on='customer_id', how='outer'):
    # Juntar los datasets
    df_merged = pd.merge(df_contract, df_personal, on=on, how=how)
    df_merged = pd.merge(df_merged, df_internet, on=on, how=how)
    df_merged = pd.merge(df_merged, df_phone, on=on, how=how)

    # Rellenar valores ausentes provocados por el merge. Se rellenaran con false.
    df_merged.fillna(False, inplace=True)

    return df_merged

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

def real_end_date(user_info):
    ''' Para las observaciones con valores NaT, se reemplazara la fecha de terminación
    por la fecha en la que acaba su contrato. 
    last_date: fecha actual'''
    last_date = '20-03-01'
    # Solo aplicar a los usuarios sin fecha de terminación
    if pd.isna(user_info['end_date']):
        type_contract = user_info['type'] # Extraer el tipo de contrato

        # Contrato mes a mes 
        # Para los que empezaron su contrato en febrero
        if (type_contract == 'Month-to-month') & (user_info['begin_date'].month == 2):
            return user_info['begin_date'] + pd.DateOffset(months=1) # Agregar un mes
        # Para los que empezaron su contrato en enero
        elif (type_contract == 'Month-to-month') & (user_info['begin_date'].month == 1):
            return pd.to_datetime('2020-02-01') # Vence en febrero
        
        # Dependiendo los años del contrato regresa una fecha
        elif type_contract == 'One year':
            return calculate_end_date_by_year(user_info, add_n_years=1)
        else:
            return calculate_end_date_by_year(user_info, add_n_years=2)
    else:
        return user_info['end_date']

def split_dates(df, date_column, prefix):
    ''' Separa la columna de fecha tipo datetime en 2 distinas columnas (mes/año).
    date_column: Columna con las fechas a dividir.
    prefix: nombre de la nueva columna. '''
    df[f'{prefix}_month'] = df[date_column].dt.month
    df[f'{prefix}_year'] = df[date_column].dt.year
    return df

def split_target_features(df, target_col):
    # Definir las características y el objetivo
    features = df.drop(target_col, axis=1)
    target = df[target_col]
    return features, target

def MinMaxScaler_dataset(features_train_encoded, features_test_encoded, columns_to_scale):
    scaler = MinMaxScaler()
    features_train_encoded_scaled = features_train_encoded.copy()
    features_test_encoded_scaled = features_test_encoded.copy()
    features_train_encoded_scaled[columns_to_scale] = scaler.fit_transform(features_train_encoded[columns_to_scale])
    features_test_encoded_scaled[columns_to_scale] = scaler.transform(features_test_encoded[columns_to_scale])
    return features_train_encoded_scaled, features_test_encoded_scaled

def LabelEncoder_dataset(features_train, features_test, features_train_encoded, features_test_encoded, lb_cols):
    label_encoder = LabelEncoder()
    features_train_encoded[lb_cols] = label_encoder.fit_transform(features_train[lb_cols])
    features_test_encoded[lb_cols] = label_encoder.transform(features_test[lb_cols])
    return features_train_encoded, features_test_encoded

def Boruta_alg(features_train_encoded, features_test_encoded, features_train_encoded_scaled, features_test_encoded_scaled, target_train):
    # Se utilizara un modelo de bosque aletorio como base para el algoritmo Boruta

    # Inicializar el modelo base de bosque aleatorio
    rf = RandomForestClassifier(random_state=54321)

    # Inicializar Boruta
    boruta_selector = BorutaPy(estimator=rf, n_estimators='auto', verbose=2)

    # Ajustar Boruta al conjunto de datos
    boruta_selector.fit(features_train_encoded.values, target_train)

    # Seleccionar las características mas relevantes de boruta
    boruta_features = features_train_encoded.columns[boruta_selector.support_]

    # Se va a filtrar el dataset con las columnas obtenidas por Boruta
    features_train_encoded = features_train_encoded[boruta_features]
    features_test_encoded = features_test_encoded[boruta_features]
    features_train_encoded_scaled = features_train_encoded_scaled[boruta_features]
    features_test_encoded_scaled = features_test_encoded_scaled[boruta_features]

    return features_train_encoded, features_test_encoded, features_train_encoded_scaled, features_test_encoded_scaled

def evaluate_model(model, train_features, train_target, test_features, test_target, proba=True):
    
    eval_stats = {}
    
    fig, axs = plt.subplots(1, 3, figsize=(20, 6)) 
    
    for type, features, target in (('train', train_features, train_target), ('test', test_features, test_target)):
        
        eval_stats[type] = {}
    
        pred_target = model.predict(features)
        pred_proba = model.predict_proba(features)[:, 1]

        # F1
        f1_thresholds = np.arange(0, 1.01, 0.05)
        f1_scores = [f1_score(target, pred_proba>=threshold) for threshold in f1_thresholds]
        
        # ROC-AUC
        fpr, tpr, roc_thresholds = roc_curve(target, pred_proba)
        roc_auc = roc_auc_score(target, pred_proba)    
        eval_stats[type]['ROC AUC'] = roc_auc

        # PRC
        precision, recall, pr_thresholds = precision_recall_curve(target, pred_proba)
        aps = average_precision_score(target, pred_proba)
        eval_stats[type]['APS'] = aps
        
        if type == 'train':
            color = 'blue'
        else:
            color = 'green'

        # Valor F1
        ax = axs[0]
        max_f1_score_idx = np.argmax(f1_scores)
        ax.plot(f1_thresholds, f1_scores, color=color, label=f'{type}, max={f1_scores[max_f1_score_idx]:.2f} @ {f1_thresholds[max_f1_score_idx]:.2f}')
        # establecer cruces para algunos umbrales        
        for threshold in (0.2, 0.4, 0.5, 0.6, 0.8):
            closest_value_idx = np.argmin(np.abs(f1_thresholds-threshold))
            marker_color = 'orange' if threshold != 0.5 else 'red'
            ax.plot(f1_thresholds[closest_value_idx], f1_scores[closest_value_idx], color=marker_color, marker='X', markersize=7)
        ax.set_xlim([-0.02, 1.02])    
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('threshold')
        ax.set_ylabel('F1')
        ax.legend(loc='lower center')
        ax.set_title(f'Valor F1') 

        # ROC
        ax = axs[1]    
        ax.plot(fpr, tpr, color=color, label=f'{type}, ROC AUC={roc_auc:.2f}')
        # establecer cruces para algunos umbrales        
        for threshold in (0.2, 0.4, 0.5, 0.6, 0.8):
            closest_value_idx = np.argmin(np.abs(roc_thresholds-threshold))
            marker_color = 'orange' if threshold != 0.5 else 'red'            
            ax.plot(fpr[closest_value_idx], tpr[closest_value_idx], color=marker_color, marker='X', markersize=7)
        ax.plot([0, 1], [0, 1], color='grey', linestyle='--')
        ax.set_xlim([-0.02, 1.02])    
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        ax.legend(loc='lower center')        
        ax.set_title(f'Curva ROC')
        
        # PRC
        ax = axs[2]
        ax.plot(recall, precision, color=color, label=f'{type}, AP={aps:.2f}')
        # establecer cruces para algunos umbrales        
        for threshold in (0.2, 0.4, 0.5, 0.6, 0.8):
            closest_value_idx = np.argmin(np.abs(pr_thresholds-threshold))
            marker_color = 'orange' if threshold != 0.5 else 'red'
            ax.plot(recall[closest_value_idx], precision[closest_value_idx], color=marker_color, marker='X', markersize=7)
        ax.set_xlim([-0.02, 1.02])    
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('recall')
        ax.set_ylabel('precision')
        ax.legend(loc='lower center')
        ax.set_title(f'PRC')        

        eval_stats[type]['Accuracy'] = accuracy_score(target, pred_target)
        eval_stats[type]['F1'] = f1_score(target, pred_target)
    
    df_eval_stats = pd.DataFrame(eval_stats)
    df_eval_stats = df_eval_stats.round(2)
    df_eval_stats = df_eval_stats.reindex(index=('Accuracy', 'F1', 'APS', 'ROC AUC'))
    
    print(df_eval_stats)