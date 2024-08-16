import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from utils.functions import evaluate_model

def dummytest(features_train, target_train, features_test, target_test, show_metrics=True):

    # Entrenamiento
    model = DummyClassifier()
    model.fit(features_train, target_train)

    # Guardar las predicciones
    predicts = pd.Series(model.predict(features_test))
    predicts.to_csv('./files/datasets/output/dummy_predicts.csv', index=False)

    if show_metrics:
        # Evaluar el modelo para Exactitud, F1, APS, ROC-AUC
        evaluate_model(model, features_train, target_train, features_test, target_test)