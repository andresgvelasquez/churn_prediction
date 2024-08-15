import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

def log_reg_model(features_train, target_train, features_test, target_test, show_metrics=False):
    # Entrenar el modelo
    model = LogisticRegression()
    model.fit(features_train, target_train)

    # Predicciones en Serie
    predicts = pd.Series(model.predict(features_test))

    # Guardar las predicciones
    predicts.to_csv('./files/datasets/output/logistic_regression_predicts.csv', index=False)

    if show_metrics:
        # Imprime las métricas ROC-AUC, F1 y Accuracy
        print(f'ROC-AUC: {roc_auc_score(target_test, predicts)}')
        print(f'F1: {f1_score(target_test, predicts)}')
        print(f'Accuracy: {accuracy_score(target_test, predicts)}')