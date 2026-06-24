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

df = df.drop(columns = ['trecho', 'tipo_de_acidente', 'hour', 'bin_hora', 'km_bin_id'])


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
Resumo dos modelos
                 model  threshold  accuracy  precision_fatal  recall_fatal  f1_fatal     tn   fp   fn   tp
2        random_forest        0.2  0.981138         0.351536      0.238426  0.284138  26894  190  329  103
0  logistic_regression        0.8  0.963258         0.185668      0.395833  0.252772  26334  750  261  171
1        decision_tree        0.5  0.975287         0.157459      0.131944  0.143577  26779  305  375   57

================================================================================
Teste 2 - Separando 'colisao'

Resumo dos modelos
                 model  threshold  accuracy  precision_fatal  recall_fatal  f1_fatal     tn    fp   fn   tp
2        random_forest        0.2  0.980920         0.346535      0.243056  0.285714  26886   198  327  105
0  logistic_regression        0.7  0.931931         0.121387      0.534722  0.197859  25412  1672  201  231
1        decision_tree        0.5  0.975214         0.160326      0.136574  0.147500  26775   309  373   59

================================================================================
Teste 3 - Separando periodos do dia
df['periodo'] = pd.cut(df['hour'], bins=[0, 6, 12, 18, 25], labels=['madrugada', 'manha', 'tarde', 'noite'], right=False, include_lowest=True)
encoded = encoder.fit_transform(df[['periodo']])
df_encoded = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['periodo']))
df = pd.concat([df, df_encoded], axis=1)
df = df.drop(columns = ['trecho', 'tipo_de_acidente', 'hour', 'bin_hora', 'periodo'])
 -> Performou pior, não vou utilizar.
 Resumo dos modelos
                 model  threshold  accuracy  precision_fatal  recall_fatal  f1_fatal     tn   fp   fn   tp
2        random_forest        0.2  0.980775         0.338870      0.236111  0.278308  26885  199  330  102
0  logistic_regression        0.8  0.961041         0.184418      0.432870  0.258645  26257  827  245  187
1        decision_tree        0.5  0.975941         0.173295      0.141204  0.155612  26793  291  371   61

================================================================================
Test 4 - Teentei manter somente o 'km_ajustado' ou somente o 'km_bin_id', e o primeiro ('km_ajustado') performou muito melhor entre eles, então vou manter, com o seguinte resultado abaixo:
Resumo dos modelos
                 model  threshold  accuracy  precision_fatal  recall_fatal  f1_fatal     tn   fp   fn   tp
2        random_forest        0.2  0.980811         0.352761      0.266204  0.303430  26873  211  317  115
0  logistic_regression        0.8  0.961622         0.186747      0.430556  0.260504  26274  810  246  186
1        decision_tree        0.5  0.975432         0.159218      0.131944  0.144304  26783  301  375   57

================================================================================
Teste 5 - ano e mes separados pelo OHE pioraram os modelos
Resumo dos modelos - ano
                 model  threshold  accuracy  precision_fatal  recall_fatal  f1_fatal     tn   fp   fn   tp
2        random_forest        0.2  0.980738         0.341935      0.245370  0.285714  26880  204  326  106
0  logistic_regression        0.8  0.960968         0.185294      0.437500  0.260331  26253  831  243  189
1        decision_tree        0.5  0.973688         0.149038      0.143519  0.146226  26730  354  370   62

Resumo dos modelos - mes
                 model  threshold  accuracy  precision_fatal  recall_fatal  f1_fatal     tn   fp   fn   tp
2        random_forest        0.2  0.980629         0.342679      0.254630  0.292165  26873  211  322  110
0  logistic_regression        0.8  0.961295         0.183183      0.423611  0.255765  26268  816  249  183
1        decision_tree        0.5  0.973761         0.131980      0.120370  0.125908  26742  342  380   52
 
"""