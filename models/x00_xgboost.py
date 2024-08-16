import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from utils.functions import evaluate_model
import pandas as pd

def xgboost_model(features_train_encoded, target_train, features_test_encoded, target_test, show_metrics=True):
    # Definir la grilla de hiperparámetros para la búsqueda en cuadrícula
    # Nota los hiperaparámetros puestos a continuación son los mejores obtenidos en la búsqueda gridsearch
    # Para minimizar el tiempo de entrenamiento, se utilizaron los mejores.
    param_grid = {
        'objective': ['binary:logistic'],
        'eval_metric': ['auc'],
        'learning_rate': [0.1],
        'max_depth': [24],
        'reg_lambda': [1],
        'n_estimators': [1000],
    }

    # Definir el modelo XGBClassifier
    xgb_classifier = xgb.XGBClassifier()

    # Realizar la búsqueda en cuadrícula con validación cruzada
    grid_search = GridSearchCV(estimator=xgb_classifier, param_grid=param_grid, cv=5, scoring='roc_auc')
    grid_search.fit(features_train_encoded.values, target_train.values)

    # Obtener los mejores hiperparámetros y el mejor modelo
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print("Mejores hiperparámetros encontrados:", best_params)

    # Guardar las predicciones
    predicts = pd.Series(best_model.predict(features_test_encoded))
    predicts.to_csv('./files/datasets/output/XGBoost_predicts.csv', index=False)

    if show_metrics:
        # Evaluar el mejor modelo para Exactitud, F1, APS, ROC-AUC
        evaluate_model(best_model, features_train_encoded, target_train, features_test_encoded, target_test)