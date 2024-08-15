import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from utils.functions import Boruta_alg, LabelEncoder_dataset, MinMaxScaler_dataset, split_target_features

def preparing_data(df_preprocessed, target_col:str='is_active', ohe_cols='payment_method', lb_cols='type',
                   columns_to_scale=['type', 'monthly_charges', 'total_charges','active_days']):
    
    # Dividir el objetivo de las características
    features, target = split_target_features(df_preprocessed, target_col)

    # Aplicar sobremuestreo para balancear el objetivo
    over_sampler = RandomOverSampler(random_state=54321)
    features_oversampled, target_oversampled = over_sampler.fit_resample(features, target)

    # Dividir el dataframe en entrenamiento y prueba
    features_train, features_test, target_train, target_test = train_test_split(
        features_oversampled, target_oversampled, test_size=0.2, random_state=54321
    )

    # One Hot Encoding (payment_method)
    features_train_encoded = pd.get_dummies(features_train, columns=[ohe_cols])
    features_test_encoded = pd.get_dummies(features_test, columns=[ohe_cols])    

    # Label encoding (Type)
    features_train_encoded, features_test_encoded = LabelEncoder_dataset(features_train, features_test, features_train_encoded, features_test_encoded, lb_cols)

    # Escalar las características de prueba y entrenamiento
    features_train_encoded_scaled, features_test_encoded_scaled = MinMaxScaler_dataset(features_train_encoded, features_test_encoded, columns_to_scale)

    # Seleccion de características importantes a partir de Boruta
    features_train_encoded, features_test_encoded, features_train_encoded_scaled, features_test_encoded_scaled = Boruta_alg(
        features_train_encoded, features_test_encoded, features_train_encoded_scaled, features_test_encoded_scaled, target_train
    )

    return features_train_encoded, features_test_encoded, features_train_encoded_scaled, features_test_encoded_scaled, target_train, target_test