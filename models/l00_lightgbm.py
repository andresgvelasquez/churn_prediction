import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from utils.functions import evaluate_model
import pandas as pd

def lgbm_model(features_train_encoded, target_train, features_test_encoded, target_test, show_metrics=True):

    # Definir la grilla de hiperparámetros para la búsqueda en cuadrícula
    # Nota los hiperaparámetros puestos a continuación son los mejores obtenidos en la búsqueda gridsearch
    # Para minimizar el tiempo de entrenamiento, se utilizaron los mejores.
    param_grid = {
        'objective': ['binary'],
        'metric': ['auc'],
        'learning_rate': [0.1],
        'max_depth': [24],
        'reg_lambda': [1],
        'n_estimators': [1000],
        'num_leaves': [24],
        'verbose_eval': [100]
    }

    # Crear un clasificador LightGBM
    lgb_classifier = lgb.LGBMClassifier()

    # Realizar la búsqueda en cuadrícula con validación cruzada
    grid_search = GridSearchCV(estimator=lgb_classifier, param_grid=param_grid, cv=5, scoring='roc_auc')
    grid_search.fit(features_train_encoded.values, target_train.values)

    # Obtener los mejores hiperparámetros y el mejor modelo
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print("Mejores hiperparámetros encontrados:", best_params)

    # Guardar las predicciones
    predicts = pd.Series(best_model.predict(features_test_encoded))
    predicts.to_csv('./files/datasets/output/lightGBM_predicts.csv', index=False)

    # Evaluar el mejor modelo para Exactitud, F1, APS, ROC-AUC
    if show_metrics:

        evaluate_model(best_model, features_train_encoded, target_train, features_test_encoded, target_test)