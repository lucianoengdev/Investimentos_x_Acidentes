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

df = pd.read_csv('../data/processed/acidentes_tratados.csv')

# Dada uma ocorrência de acidente, conseguimos prever se ela será fatal?

# Colunas a manter = [['horario', 'trecho', 'sentido', 'tipo_de_acidente', 'automovel', 'bicicleta',  'caminhao', 'moto', 'onibus', 'outros', 'tracao_animal', 'transporte_de_cargas_especiais', 'trator_maquinas', 'utilitarios', 'km_ajustado', 'ano', 'mes', 'acidente_fatal', 'km_bin']] 
# obs. vou largar o km_bin e criar outro porque ele vem como texto e quero utiliza-lo categoricamente
# Colunas a excluir = [['data', 'km', 'n_da_ocorrencia', 'ilesos', 'levemente_feridos', 'moderadamente_feridos', 'gravemente_feridos', 'mortos']]


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

# OneHotEncoder para trecho e tipo de acidente
encoder = OneHotEncoder(sparse_output=False)
encoded = encoder.fit_transform(df[['trecho']])
df_encoded = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(["trecho"]))
df = pd.concat([df, df_encoded], axis=1)

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
df.loc[df['tipo_de_acidente'].str.contains('colisao'), 'tipo_de_acidente'] = 'colisao'
manter_atropelamento = ['atropelamento_de_pedestre', 'atropelamento_morador', 'atropelamento_andarilho']
df.loc[df['tipo_de_acidente'].str.contains('atropelamento') & ~df['tipo_de_acidente'].isin(manter_atropelamento), 'tipo_de_acidente'] = 'atropelamento_outros'
manter = ['atropelamento_de_pedestre', 'atropelamento_morador', 'atropelamento_andarilho', 'atropelamento_outros', 'choque', 'colisao', 'tombamento', 'engavetamento', 'capotamento']
df.loc[~df['tipo_de_acidente'].isin(manter), 'tipo_de_acidente'] = 'outros'

encoded = encoder.fit_transform(df[['tipo_de_acidente']])
df_encoded = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['tipo_de_acidente']))
df = pd.concat([df, df_encoded], axis=1)
df = df.drop(columns = ['trecho', 'tipo_de_acidente'])

X = df.drop(columns = 'acidente_fatal')
y = df['acidente_fatal']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = 42)


fig_dir = Path('../reports/figures/04_modelos')
fig_dir.mkdir(parents=True, exist_ok=True)
"""dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(X_train, y_train)
y_pred_dummy = dummy.predict(X_test)
accuracy_test = accuracy_score(y_test, y_pred_dummy)

print("Acurácia Dummy:", accuracy_test)
print("Distribuição y_test:")
print(y_test.value_counts(normalize=True))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred_dummy,
    display_labels=['Nao fatal', 'Fatal'],
    cmap='Blues'
)
plt.title('Matriz de confusao - DummyClassifier most_frequent')
plt.tight_layout()
plt.savefig(fig_dir / 'dummy_confusion_matrix.png')
plt.close()

distribuicao_dummy = pd.DataFrame({
    'Real': y_test.value_counts(normalize=True).sort_index(),
    'Previsto pelo Dummy': pd.Series(y_pred_dummy).value_counts(normalize=True).sort_index()
}).fillna(0)

distribuicao_dummy.plot(kind='bar', figsize=(8, 5))
plt.title('Distribuicao real x previsao Dummy')
plt.xlabel('Classe')
plt.ylabel('Proporcao')
plt.xticks(ticks=[0, 1], labels=['Nao fatal', 'Fatal'], rotation=0)
plt.tight_layout()
plt.savefig(fig_dir / 'dummy_class_distribution.png')
plt.close()

Acurácia Dummy: 0.98430004361099
Distribuição y_test:
acidente_fatal
0    0.9843
1    0.0157
Name: proportion, dtype: float64
"""

col_normalizar = ['km_ajustado', 'ano']

scaler = MinMaxScaler()

X_train[col_normalizar] = scaler.fit_transform(X_train[col_normalizar])
X_test[col_normalizar] = scaler.transform(X_test[col_normalizar])


"""
model = LogisticRegression(max_iter = 1000, class_weight='balanced')
model.fit(X_train, y_train)


y_proba_fatal = model.predict_proba(X_test)[:,1]

thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

precisions = []
recalls = []
f1_scores = []

for threshold in thresholds:
    y_pred_threshold = (y_proba_fatal >= threshold).astype(int)

    precisions.append(precision_score(y_test, y_pred_threshold))
    recalls.append(recall_score(y_test, y_pred_threshold))
    f1_scores.append(f1_score(y_test, y_pred_threshold))

plt.figure(figsize=(10,6))
plt.plot(thresholds, precisions, marker = 'o', label = 'Precision')
plt.plot(thresholds, recalls, marker = 'o', label = 'Recall')
plt.plot(thresholds, f1_scores, marker = 'o', label = 'F1-Score')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Precision, Recall e F1 por threshold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(fig_dir / 'precision_recall_f1.png')

thresholds_cm = [0.5, 0.6, 0.7, 0.8]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for threshold, ax in zip(thresholds_cm, axes.flatten()):
    y_pred_threshold = (y_proba_fatal >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred_threshold)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Nao fatal', 'Fatal'])
    disp.plot(ax=ax, cmap='Blues', values_format='d', colorbar=False)
    ax.set_title(f'Threshold {threshold}')
plt.suptitle('Matrizes de confusao - Logistic Regression')
plt.tight_layout()
plt.savefig(fig_dir / 'confusion_matrix_logreg_thresholds.png')
 

Teste 1 - Logistic Regression                               # y_log_pred = model.predict(X_test)
Acurácia: 0.800770460822794
[[21713  5371]
 [  111   321]]
              precision    recall  f1-score   support

           0       0.99      0.80      0.89     27084
           1       0.06      0.74      0.10       432

    accuracy                           0.80     27516
   macro avg       0.53      0.77      0.50     27516
weighted avg       0.98      0.80      0.88     27516

Teste 2 - thresholds
thresholds:  0.3
-------------------------
Acurácia: 0.5912559965111208
[[15873 11211]
 [   36   396]]
              precision    recall  f1-score   support

           0       1.00      0.59      0.74     27084
           1       0.03      0.92      0.07       432

    accuracy                           0.59     27516
   macro avg       0.52      0.75      0.40     27516
weighted avg       0.98      0.59      0.73     27516

thresholds:  0.4
-------------------------
Acurácia: 0.7075156272714057
[[19109  7975]
 [   73   359]]
              precision    recall  f1-score   support

           0       1.00      0.71      0.83     27084
           1       0.04      0.83      0.08       432

    accuracy                           0.71     27516
   macro avg       0.52      0.77      0.45     27516
weighted avg       0.98      0.71      0.81     27516

thresholds:  0.5
-------------------------
Acurácia: 0.800770460822794
[[21713  5371]
 [  111   321]]
              precision    recall  f1-score   support

           0       0.99      0.80      0.89     27084
           1       0.06      0.74      0.10       432

    accuracy                           0.80     27516
   macro avg       0.53      0.77      0.50     27516
weighted avg       0.98      0.80      0.88     27516

thresholds:  0.6
-------------------------
Acurácia: 0.872837621747347
[[23740  3344]
 [  155   277]]
              precision    recall  f1-score   support

           0       0.99      0.88      0.93     27084
           1       0.08      0.64      0.14       432

    accuracy                           0.87     27516
   macro avg       0.54      0.76      0.53     27516
weighted avg       0.98      0.87      0.92     27516

thresholds:  0.7
-------------------------
Acurácia: 0.9233900276202937
[[25171  1913]
 [  195   237]]
              precision    recall  f1-score   support

           0       0.99      0.93      0.96     27084
           1       0.11      0.55      0.18       432

    accuracy                           0.92     27516
   macro avg       0.55      0.74      0.57     27516
weighted avg       0.98      0.92      0.95     27516

thresholds:  0.8
-------------------------
Acurácia: 0.9519188835586568
[[26011  1073]
 [  250   182]]
              precision    recall  f1-score   support

           0       0.99      0.96      0.98     27084
           1       0.15      0.42      0.22       432

    accuracy                           0.95     27516
   macro avg       0.57      0.69      0.60     27516
weighted avg       0.98      0.95      0.96     27516

A separação dos tipos de atropelamento melhorou a qualidade do modelo, principalmente no threshold 0.8, reduzindo falsos positivos e aumentando o F1-score da classe fatal. Isso confirma que a engenharia de features baseada na análise exploratória pode melhorar a capacidade do modelo de diferenciar acidentes fatais e não fatais. Segue resultado abaixo:

thresholds:  0.3
-------------------------
Acurácia: 0.592891408634976
[[15918 11166]
 [   36   396]]
              precision    recall  f1-score   support

           0       1.00      0.59      0.74     27084
           1       0.03      0.92      0.07       432

    accuracy                           0.59     27516
   macro avg       0.52      0.75      0.40     27516
weighted avg       0.98      0.59      0.73     27516

thresholds:  0.4
-------------------------
Acurácia: 0.7078063672045355
[[19115  7969]
 [   71   361]]
              precision    recall  f1-score   support

           0       1.00      0.71      0.83     27084
           1       0.04      0.84      0.08       432

    accuracy                           0.71     27516
   macro avg       0.52      0.77      0.45     27516
weighted avg       0.98      0.71      0.81     27516

thresholds:  0.5
-------------------------
Acurácia: 0.8013519406890537
[[21729  5355]
 [  111   321]]
              precision    recall  f1-score   support

           0       0.99      0.80      0.89     27084
           1       0.06      0.74      0.11       432

    accuracy                           0.80     27516
   macro avg       0.53      0.77      0.50     27516
weighted avg       0.98      0.80      0.88     27516

thresholds:  0.6
-------------------------
Acurácia: 0.8740005814798663
[[23769  3315]
 [  152   280]]
              precision    recall  f1-score   support

           0       0.99      0.88      0.93     27084
           1       0.08      0.65      0.14       432

    accuracy                           0.87     27516
   macro avg       0.54      0.76      0.54     27516
weighted avg       0.98      0.87      0.92     27516

thresholds:  0.7
-------------------------
Acurácia: 0.9277147841255996
[[25295  1789]
 [  200   232]]
              precision    recall  f1-score   support

           0       0.99      0.93      0.96     27084
           1       0.11      0.54      0.19       432

    accuracy                           0.93     27516
   macro avg       0.55      0.74      0.58     27516
weighted avg       0.98      0.93      0.95     27516

thresholds:  0.8
-------------------------
Acurácia: 0.961404273877017
[[26277   807]
 [  255   177]]
              precision    recall  f1-score   support

           0       0.99      0.97      0.98     27084
           1       0.18      0.41      0.25       432

    accuracy                           0.96     27516
   macro avg       0.59      0.69      0.62     27516
weighted avg       0.98      0.96      0.97     27516
"""

tree_model = DecisionTreeClassifier(random_state=42, class_weight='balanced')

tree_model.fit(X_train, y_train)
valuesof_tree_model = tree_model.score(X_test, y_test)
print(valuesof_tree_model)

y_tree_pred = tree_model.predict(X_test)

print(confusion_matrix(y_test, y_tree_pred))
print(classification_report(y_test, y_tree_pred))

importances = tree_model.feature_importances_
for name, imp in zip(X.columns, importances):
   print(name, imp)

feat_importance = pd.DataFrame({'feature': X.columns, 'importance': importances})
feat_importance = feat_importance.sort_values(by='importance', ascending=False)
print(feat_importance)

max_depths = [4, 6, 8, 10]
min_samples_leafs = [20, 50, 100]
tree_results = []

for max_depth in max_depths:
    for min_samples_leaf in min_samples_leafs:
        tree_test = DecisionTreeClassifier(
            random_state=42,
            class_weight='balanced',
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf
        )

        tree_test.fit(X_train, y_train)
        y_train_pred_tree = tree_test.predict(X_train)
        y_test_pred_tree = tree_test.predict(X_test)

        tree_results.append({
            'max_depth': max_depth,
            'min_samples_leaf': min_samples_leaf,
            'train_accuracy': accuracy_score(y_train, y_train_pred_tree),
            'test_accuracy': accuracy_score(y_test, y_test_pred_tree),
            'train_precision_fatal': precision_score(y_train, y_train_pred_tree, zero_division=0),
            'test_precision_fatal': precision_score(y_test, y_test_pred_tree, zero_division=0),
            'train_recall_fatal': recall_score(y_train, y_train_pred_tree, zero_division=0),
            'test_recall_fatal': recall_score(y_test, y_test_pred_tree, zero_division=0),
            'train_f1_fatal': f1_score(y_train, y_train_pred_tree, zero_division=0),
            'test_f1_fatal': f1_score(y_test, y_test_pred_tree, zero_division=0)
        })

tree_results_df = pd.DataFrame(tree_results)
print(tree_results_df)

plt.figure(figsize=(12, 6))
for min_samples_leaf in min_samples_leafs:
    df_leaf = tree_results_df[tree_results_df['min_samples_leaf'] == min_samples_leaf]
    plt.plot(
        df_leaf['max_depth'],
        df_leaf['train_f1_fatal'],
        marker='o',
        linestyle='--',
        label=f'Treino F1 fatal - leaf {min_samples_leaf}'
    )
    plt.plot(
        df_leaf['max_depth'],
        df_leaf['test_f1_fatal'],
        marker='o',
        linestyle='-',
        label=f'Teste F1 fatal - leaf {min_samples_leaf}'
    )

plt.xlabel('max_depth')
plt.ylabel('F1-score da classe fatal')
plt.title('Decision Tree - F1 fatal por profundidade e min_samples_leaf')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(fig_dir / 'decision_tree_f1_overfitting_curve.png')
plt.close()

plt.figure(figsize=(12, 6))
for min_samples_leaf in min_samples_leafs:
    df_leaf = tree_results_df[tree_results_df['min_samples_leaf'] == min_samples_leaf]
    plt.plot(
        df_leaf['max_depth'],
        df_leaf['test_precision_fatal'],
        marker='o',
        label=f'Precision fatal - leaf {min_samples_leaf}'
    )
    plt.plot(
        df_leaf['max_depth'],
        df_leaf['test_recall_fatal'],
        marker='x',
        linestyle='--',
        label=f'Recall fatal - leaf {min_samples_leaf}'
    )

plt.xlabel('max_depth')
plt.ylabel('Score no teste')
plt.title('Decision Tree - Precision e Recall fatal no teste')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(fig_dir / 'decision_tree_precision_recall_curve.png')
plt.close()
