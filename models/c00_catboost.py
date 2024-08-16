import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from utils.functions import evaluate_model

def catboost_model(features_train, target_train, features_test, target_test, show_metrics=True):

    # Definir la grilla de hiperparámetros para la búsqueda aleatoria
    # *Nota: La busueda de hiperparametros ya se realizo, por lo que los hiperaparámetros puestos a continuación son los que obtienen un mejor resultado.
    # Esto con el objetivo de reducir el tiempo del modelo.

    param_grid = {
        'learning_rate': [0.1],
        'depth': [6],
        'l2_leaf_reg': [1],
        'iterations': [500],
        'verbose_eval': [100]
    }

    # Métricas de evaluación
    scoring = {'ROC-AUC': 'roc_auc', 'F1': 'f1', 'Accuracy': 'accuracy'}

    # Definir el modelo catboost
    catboost = CatBoostClassifier()

    # Realizar la búsqueda aleatoria con validación cruzada
    grid_search = GridSearchCV(estimator=catboost, param_grid=param_grid, cv=5, scoring=scoring, refit='ROC-AUC', verbose=2)
    grid_search.fit(features_train, target_train.astype(int))

    # Obtener los mejores hiperparámetros
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print("Mejores hiperparámetros encontrados:", best_params)
    
    # Guardar las predicciones
    predicts = pd.Series(best_model.predict(features_test))
    predicts.to_csv('./files/datasets/output/catboost_predicts.csv', index=False)

    if show_metrics:
        # Evaluar el mejor modelo para Exactitud, F1, APS, ROC-AUC
        evaluate_model(best_model, features_train, target_train, features_test, target_test)