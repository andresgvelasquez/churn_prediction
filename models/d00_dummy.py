import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

def dummytest(features_train, target_train, features_test, target_test, show_metrics=False):
    #Entrenamiento
    model = DummyClassifier()
    model.fit(features_train, target_train)

    # Predicciones
    predicts = pd.Series(model.predict(features_test))

    # Guardar las predicciones
    predicts.to_csv('./files/datasets/output/dummy_predicts.csv', index=False)

    if show_metrics:
        # Imprime las métricas ROC-AUC, F1 y Accuracy
        print(f'ROC-AUC: {roc_auc_score(target_test, predicts)}')
        print(f'F1: {f1_score(target_test, predicts)}')
        print(f'Accuracy: {accuracy_score(target_test, predicts)}')