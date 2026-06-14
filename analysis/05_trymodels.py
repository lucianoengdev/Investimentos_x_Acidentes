import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('../data/processed/acidentes_tratados.csv')

col_trash = ['data', 'km', 'tipo_de_ocorrencia', 'n_da_ocorrencia', 'ilesos', 'levemente_feridos', 'moderadamente_feridos', 'gravemente_feridos', 'mortos', 'km_bin']
df = df.drop(col_trash, axis=1)
df['tipo_de_acidente'] = (df['tipo_de_acidente'].replace('', 'Não informado').fillna('Não informado'))
bin20 = range(0, int(df['km_ajustado'].max()) + 20, 20)
df['km_bin_id'] = pd.cut(df['km_ajustado'], bins=bin20, include_lowest=True, labels=False)

df['horario'] = pd.to_datetime(df['horario'], format='%H:%M:%S')
minutos = (df['horario'].dt.hour * 60 + df['horario'].dt.minute)
df['hora_sin'] = np.sin(2 * np.pi * minutos / 1440)
df['hora_cos'] = np.cos(2 * np.pi * minutos / 1440)

df['hour'] = df['horario'].dt.hour
df['bin_hora'] = pd.cut(df['hour'], bins=range(0, 25), labels=False, right=False)
df = df.drop(columns='horario')

# SUL = 0 / NORTE = 1 
df['sentido'] = df['sentido'].replace({'Sul': 0, 'Crescente': 0, 'Norte': 1, 'Decrescente': 1})
# df_try = ((df['sentido'] != 0) & (df['sentido'] != 1)).sum() -> Confirmei se sobrou coluna diferente e nao sobrou

# OneHotEncoder para tipo de acidente
encoder = OneHotEncoder(sparse_output=False)

df['tipo_de_acidente'] = (
    df['tipo_de_acidente']
    .str.lower()
    .str.strip()
    .str.normalize('NFKD')
    .str.encode('ascii', errors='ignore')
    .str.decode('utf-8')
    .str.replace('-', '_', regex=False)
    .str.replace(r'\s+', '_', regex=True)
    .str.replace(r'_+', '_', regex=True)
)
# df.loc[df['tipo_de_acidente'].str.contains('atropelamento'), 'tipo_de_acidente'] = 'atropelamento'
df.loc[df['tipo_de_acidente'].str.contains('choque'), 'tipo_de_acidente'] = 'choque'
df.loc[df['tipo_de_acidente'].str.contains('colisao_lateral'), 'tipo_de_acidente'] = 'colisao_lateral'
manter_atropelamento = ['atropelamento_de_pedestre', 'atropelamento_morador', 'atropelamento_andarilho']
df.loc[df['tipo_de_acidente'].str.contains('atropelamento') & ~df['tipo_de_acidente'].isin(manter_atropelamento), 'tipo_de_acidente'] = 'atropelamento_outros'
manter = ['atropelamento_de_pedestre', 'atropelamento_morador', 'atropelamento_andarilho', 'atropelamento_outros', 'choque', 'colisao', 'tombamento', 'engavetamento', 'capotamento', 'colisao_lateral', 'colisao_frontal', 'colisao_traseira', 'colisao_transversal']
df.loc[~df['tipo_de_acidente'].isin(manter), 'tipo_de_acidente'] = 'outros'

encoded = encoder.fit_transform(df[['tipo_de_acidente']])
df_encoded = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['tipo_de_acidente']))
df = pd.concat([df, df_encoded], axis=1)
df = df.drop(columns = ['trecho', 'tipo_de_acidente', 'hour', 'bin_hora'])

X = df.drop(columns = 'acidente_fatal')
y = df['acidente_fatal']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = 42)


models = {
    'logistic_regression': {
        'model': LogisticRegression(
            random_state=42,
            class_weight='balanced',
            max_iter=1000
        ),
        'threshold': 0.8,
        'requires_scaling': True,
    },

    'decision_tree': {
        'model': DecisionTreeClassifier(
            random_state=42,
            class_weight='balanced'
        ),
        'threshold': 0.5,
        'requires_scaling': False,
    },

    'random_forest': {
        'model': RandomForestClassifier(
            random_state=42,
            class_weight='balanced',
            n_estimators=100,
            max_depth=None,
            min_samples_leaf=1,
            max_features='sqrt',
            n_jobs=-1
        ),
        'threshold': 0.2,
        'requires_scaling': False,
    }
}

results = []

for model_name, config in models.items():
    model = config['model']
    threshold = config['threshold']

    if config['requires_scaling']:
        scaler = MinMaxScaler()
        X_train_model = scaler.fit_transform(X_train)
        X_test_model = scaler.transform(X_test)
    else:
        X_train_model = X_train
        X_test_model = X_test

    model.fit(X_train_model, y_train)

    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test_model)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)
    else:
        y_pred = model.predict(X_test_model)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    results.append({
        'model': model_name,
        'threshold': threshold,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision_fatal': precision_score(y_test, y_pred, zero_division=0),
        'recall_fatal': recall_score(y_test, y_pred, zero_division=0),
        'f1_fatal': f1_score(y_test, y_pred, zero_division=0),
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp
    })

    print('\n' + '=' * 80)
    print(model_name)
    print(f"Threshold: {threshold}")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, zero_division=0))

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by='f1_fatal', ascending=False)

print('\n' + '=' * 80)
print('Resumo dos modelos')
print(results_df)

print(df.columns)

"""
Teste 1 - Retirando 'trecho'
================================================================================
logistic_regression
Threshold: 0.8
[[26334   750]
 [  261   171]]
              precision    recall  f1-score   support

           0       0.99      0.97      0.98     27084
           1       0.19      0.40      0.25       432

    accuracy                           0.96     27516
   macro avg       0.59      0.68      0.62     27516
weighted avg       0.98      0.96      0.97     27516


================================================================================
decision_tree
Threshold: 0.5
[[26779   305]
 [  375    57]]
              precision    recall  f1-score   support

           0       0.99      0.99      0.99     27084
           1       0.16      0.13      0.14       432

    accuracy                           0.98     27516
   macro avg       0.57      0.56      0.57     27516
weighted avg       0.97      0.98      0.97     27516


================================================================================
random_forest
Threshold: 0.2
[[26894   190]
 [  329   103]]
              precision    recall  f1-score   support

           0       0.99      0.99      0.99     27084
           1       0.35      0.24      0.28       432

    accuracy                           0.98     27516
   macro avg       0.67      0.62      0.64     27516
weighted avg       0.98      0.98      0.98     27516


================================================================================
Resumo dos modelos
                 model  threshold  accuracy  precision_fatal  recall_fatal  f1_fatal     tn   fp   fn   tp
2        random_forest        0.2  0.981138         0.351536      0.238426  0.284138  26894  190  329  103
0  logistic_regression        0.8  0.963258         0.185668      0.395833  0.252772  26334  750  261  171
1        decision_tree        0.5  0.975287         0.157459      0.131944  0.143577  26779  305  375   57
(venv) 
"""