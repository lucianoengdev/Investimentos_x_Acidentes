import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.utils import resample


df = pd.read_csv('../data/processed/acidentes_tratados.csv')

# Objetivo deste arquivo:
# aprofundar apenas Random Forest usando a melhor estrutura de features encontrada no 05_trymodels.py.

# 1. Feature engineering base

col_trash = [
    'data',
    'km',
    'tipo_de_ocorrencia',
    'n_da_ocorrencia',
    'ilesos',
    'levemente_feridos',
    'moderadamente_feridos',
    'gravemente_feridos',
    'mortos',
    'km_bin',
]

df = df.drop(col_trash, axis=1)
df['tipo_de_acidente'] = df['tipo_de_acidente'].replace('', 'Nao informado').fillna('Nao informado')

df['horario'] = pd.to_datetime(df['horario'], format='%H:%M:%S')
minutos = df['horario'].dt.hour * 60 + df['horario'].dt.minute
df['hora_sin'] = np.sin(2 * np.pi * minutos / 1440)
df['hora_cos'] = np.cos(2 * np.pi * minutos / 1440)
df = df.drop(columns='horario')

# SUL = 0 / NORTE = 1
df['sentido'] = df['sentido'].replace({
    'Sul': 0,
    'Crescente': 0,
    'Norte': 1,
    'Decrescente': 1,
})

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

# Mesma logica vencedora do 05_trymodels.py:
# separar colisoes especificas, manter atropelamentos importantes e agrupar o restante.
df.loc[df['tipo_de_acidente'].str.contains('choque'), 'tipo_de_acidente'] = 'choque'
df.loc[df['tipo_de_acidente'].str.contains('colisao_lateral'), 'tipo_de_acidente'] = 'colisao_lateral'

manter_atropelamento = [
    'atropelamento_de_pedestre',
    'atropelamento_morador',
    'atropelamento_andarilho',
]

df.loc[
    df['tipo_de_acidente'].str.contains('atropelamento')
    & ~df['tipo_de_acidente'].isin(manter_atropelamento),
    'tipo_de_acidente',
] = 'atropelamento_outros'

manter = [
    'atropelamento_de_pedestre',
    'atropelamento_morador',
    'atropelamento_andarilho',
    'atropelamento_outros',
    'choque',
    'colisao',
    'tombamento',
    'engavetamento',
    'capotamento',
    'colisao_lateral',
    'colisao_frontal',
    'colisao_traseira',
    'colisao_transversal',
]

df.loc[~df['tipo_de_acidente'].isin(manter), 'tipo_de_acidente'] = 'outros'

encoded = encoder.fit_transform(df[['tipo_de_acidente']])
df_encoded = pd.DataFrame(
    encoded,
    columns=encoder.get_feature_names_out(['tipo_de_acidente']),
    index=df.index,
)

df = pd.concat([df, df_encoded], axis=1)

# Mantendo a feature set vencedora ate aqui:
# sem trecho, sem km_bin_id, com km_ajustado continuo, ano/mes numericos e hora ciclica.
df = df.drop(columns=['trecho', 'tipo_de_acidente', 'km_bin_id'], errors='ignore')

X = df.drop(columns='acidente_fatal')
y = df['acidente_fatal']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42,
)


# 2. Funcoes auxiliares

def apply_sampling(X_train_base, y_train_base, config):
    """Aplica undersampling apenas quando o modelo pedir."""
    if config.get('sampling') != 'undersample_majority':
        return X_train_base, y_train_base

    train_config = pd.concat([X_train_base, y_train_base], axis=1)
    df_majority = train_config[train_config['acidente_fatal'] == 0]
    df_minority = train_config[train_config['acidente_fatal'] == 1]

    df_majority_downsampled = resample(
        df_majority,
        replace=False,
        n_samples=min(
            len(df_majority),
            len(df_minority) * config['majority_ratio']
        ),
        random_state=42,
    )

    train_config = pd.concat([df_majority_downsampled, df_minority])
    train_config = train_config.sample(frac=1, random_state=42)

    return (
        train_config.drop(columns='acidente_fatal'),
        train_config['acidente_fatal'],
    )


def evaluate_predictions(model_name, threshold, y_true, y_proba):
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        'model': model_name,
        'threshold': threshold,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_fatal': precision_score(y_true, y_pred, zero_division=0),
        'recall_fatal': recall_score(y_true, y_pred, zero_division=0),
        'f1_fatal': f1_score(y_true, y_pred, zero_division=0),
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp,
    }


# 3. Random Forest profundo
# Aqui ficam apenas variacoes de Random Forest.
# A diferenca em relacao ao 05_trymodels.py e que este arquivo testa
# hiperparametros, undersampling e thresholds de maneira mais aprofundada.
models = {
    'rf_conservative_reference': {
        'model': RandomForestClassifier(
            random_state=42,
            class_weight='balanced',
            n_estimators=300,
            min_samples_leaf=2,
            max_features='sqrt',
            n_jobs=-1,
        ),
        'thresholds': [0.33, 0.34, 0.35, 0.36, 0.37],
    },

    'rf_undersample_15x_reference': {
        'model': RandomForestClassifier(
            random_state=42,
            n_estimators=300,
            min_samples_leaf=2,
            max_features='sqrt',
            n_jobs=-1,
        ),
        'thresholds': [0.35, 0.354, 0.356, 0.36, 0.37],
        'sampling': 'undersample_majority',
        'majority_ratio': 15,
    },

    'rf_undersample_10x_more_recall': {
        'model': RandomForestClassifier(
            random_state=42,
            n_estimators=300,
            min_samples_leaf=2,
            max_features='sqrt',
            n_jobs=-1,
        ),
        'thresholds': [0.40, 0.42, 0.44, 0.46, 0.48],
        'sampling': 'undersample_majority',
        'majority_ratio': 10,
    },

    'rf_undersample_20x_more_precision': {
        'model': RandomForestClassifier(
            random_state=42,
            n_estimators=300,
            min_samples_leaf=2,
            max_features='sqrt',
            n_jobs=-1,
        ),
        'thresholds': [0.27, 0.29, 0.30, 0.31, 0.33],
        'sampling': 'undersample_majority',
        'majority_ratio': 20,
    },

    'rf_leaf3_test': {
        'model': RandomForestClassifier(
            random_state=42,
            class_weight='balanced',
            n_estimators=500,
            min_samples_leaf=3,
            max_features='sqrt',
            n_jobs=-1,
        ),
        'thresholds': [0.42, 0.43, 0.44, 0.45],
    },
}


results = []
fitted_models = {}

for model_name, config in models.items():
    X_train_config, y_train_config = apply_sampling(X_train, y_train, config)

    model = config['model']
    model.fit(X_train_config, y_train_config)
    y_proba = model.predict_proba(X_test)[:, 1]

    fitted_models[model_name] = model

    for threshold in config['thresholds']:
        results.append(
            evaluate_predictions(
                model_name=model_name,
                threshold=threshold,
                y_true=y_test,
                y_proba=y_proba,
            )
        )

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by=['f1_fatal', 'recall_fatal', 'precision_fatal'],
    ascending=False,
)

print('\n' + '=' * 80)
print('Resumo Random Forest - melhores configuracoes')
print(results_df.to_string(index=False))


# 4. Melhor modelo

best_result = results_df.iloc[0]
best_model_name = best_result['model']
best_threshold = best_result['threshold']
best_model = fitted_models[best_model_name]
best_proba = best_model.predict_proba(X_test)[:, 1]
best_pred = (best_proba >= best_threshold).astype(int)

print('\n' + '=' * 80)
print('Melhor Random Forest encontrada')
print(best_result.to_string())
print('\nMatriz de confusao:')
print(confusion_matrix(y_test, best_pred))
print('\nClassification report:')
print(classification_report(y_test, best_pred, zero_division=0))


# 5. Importancia das features

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': best_model.feature_importances_,
})

feature_importance = feature_importance.sort_values(
    by='importance',
    ascending=False,
)

print('\n' + '=' * 80)
print('Top 20 feature_importances do melhor modelo')
print(feature_importance.head(20).to_string(index=False))


# Permutation importance e mais lenta, entao fica limitada para nao travar o fluxo.
# Ela ajuda a verificar quais features realmente alteram a performance no conjunto de teste.
perm_importance = permutation_importance(
    best_model,
    X_test,
    y_test,
    n_repeats=5,
    random_state=42,
    scoring='f1',
    n_jobs=-1,
)

perm_df = pd.DataFrame({
    'feature': X.columns,
    'permutation_importance_mean': perm_importance.importances_mean,
    'permutation_importance_std': perm_importance.importances_std,
})

perm_df = perm_df.sort_values(
    by='permutation_importance_mean',
    ascending=False,
)

print('\n' + '=' * 80)
print('Top 20 permutation importances')
print(perm_df.head(20).to_string(index=False))


# 6. Grafico simples

plt.figure(figsize=(12, 6))
plt.barh(
    feature_importance.head(15)['feature'][::-1],
    feature_importance.head(15)['importance'][::-1],
)
plt.title('Top 15 features - Random Forest')
plt.xlabel('Importancia')
plt.tight_layout()
plt.savefig('../reports/figures/04_modelos/random_forest_feature_importance.png')
plt.close()

"""
================================================================================
Melhor Random Forest encontrada
model              rf_undersample_15x_reference
threshold                                 0.356
accuracy                               0.978558
precision_fatal                        0.329004
recall_fatal                           0.351852
f1_fatal                               0.340045
tn                                        26774
fp                                          310
fn                                          280
tp                                          152

Matriz de confusao:
[[26774   310]
 [  280   152]]

Classification report:
              precision    recall  f1-score   support

           0       0.99      0.99      0.99     27084
           1       0.33      0.35      0.34       432

    accuracy                           0.98     27516
   macro avg       0.66      0.67      0.66     27516
weighted avg       0.98      0.98      0.98     27516

"""