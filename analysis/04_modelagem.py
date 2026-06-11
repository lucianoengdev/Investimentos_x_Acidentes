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

"""

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

02 - Tree Model tests

0.9750327082424771
[[26766   318]
 [  369    63]]
              precision    recall  f1-score   support

           0       0.99      0.99      0.99     27084
           1       0.17      0.15      0.15       432

    accuracy                           0.98     27516
   macro avg       0.58      0.57      0.57     27516
weighted avg       0.97      0.98      0.97     27516

                                       feature  importance
11                                 km_ajustado    0.175351
16                                    hora_cos    0.147809
15                                    hora_sin    0.102788
27                     tipo_de_acidente_choque    0.095411
12                                         ano    0.069140
13                                         mes    0.066325
3                                     caminhao    0.047030
4                                         moto    0.043270
1                                    automovel    0.036393
24      tipo_de_acidente_atropelamento_morador    0.033901
28                    tipo_de_acidente_colisao    0.024618
22    tipo_de_acidente_atropelamento_andarilho    0.021249
25       tipo_de_acidente_atropelamento_outros    0.016553
23  tipo_de_acidente_atropelamento_de_pedestre    0.014671
6                                       outros    0.011941
14                                   km_bin_id    0.011844
10                                 utilitarios    0.010220
29              tipo_de_acidente_engavetamento    0.009811
0                                      sentido    0.008867
5                                       onibus    0.008796
26                tipo_de_acidente_capotamento    0.008011
18                                    bin_hora    0.007734
17                                        hour    0.007265
31                 tipo_de_acidente_tombamento    0.004924
2                                    bicicleta    0.004631
30                     tipo_de_acidente_outros    0.004389
8               transporte_de_cargas_especiais    0.003868
19                            trecho_BR-381/MG    0.001487
20                            trecho_BR-381/SP    0.000978
7                                tracao_animal    0.000707
21                            trecho_Contorno/    0.000018
9                              trator_maquinas    0.000000
    max_depth  min_samples_leaf  train_accuracy  test_accuracy  train_precision_fatal  test_precision_fatal  train_recall_fatal  test_recall_fatal  train_f1_fatal  test_f1_fatal
0           4                20        0.629148       0.633014               0.032977              0.032139            0.798149           0.768519        0.063338       0.061699
1           4                50        0.628648       0.632723               0.032934              0.032386            0.798149           0.775463        0.063258       0.062175
2           4               100        0.628648       0.632723               0.032934              0.032386            0.798149           0.775463        0.063258       0.062175
3           6                20        0.673769       0.676843               0.038162              0.036692            0.816657           0.775463        0.072917       0.070069
4           6                50        0.662212       0.664850               0.037208              0.036001            0.824176           0.789352        0.071202       0.068861
5           6               100        0.662466       0.665540               0.037139              0.035975            0.821862           0.787037        0.071066       0.068805
6           8                20        0.765128       0.763846               0.051504              0.046772            0.801041           0.724537        0.096785       0.087872
7           8                50        0.759386       0.758613               0.050387              0.046583            0.802198           0.738426        0.094818       0.087637
8           8               100        0.757351       0.757886               0.049688              0.046447            0.796992           0.738426        0.093544       0.087397
9          10                20        0.834563       0.831444               0.070121              0.060042            0.777328           0.664352        0.128637       0.110130
10         10                50        0.813574       0.810983               0.063635              0.055379            0.792366           0.687500        0.117809       0.102502
11         10               100        0.786144       0.784816               0.056964              0.050450            0.810873           0.712963        0.106450       0.094233


"""

model_random_forest = RandomForestClassifier(random_state = 42, class_weight = 'balanced')
model_random_forest.fit(X_train, y_train)

"""y_random_pred = model_random_forest.predict(X_test)

print("Acurácia:", accuracy_score(y_test, y_random_pred))
print(confusion_matrix(y_test, y_random_pred))
print(classification_report(y_test, y_random_pred))

rf_thresholds = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
rf_class_weights = ['balanced', 'balanced_subsample']
rf_max_depths = [8, 12, None]
rf_min_samples_leafs = [1, 20, 50]
rf_results = []

for class_weight in rf_class_weights:
    for max_depth in rf_max_depths:
        for min_samples_leaf in rf_min_samples_leafs:
            rf_test = RandomForestClassifier(
                random_state=42,
                class_weight=class_weight,
                n_estimators=100,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                max_features='sqrt',
                n_jobs=-1
            )

            rf_test.fit(X_train, y_train)
            y_rf_proba = rf_test.predict_proba(X_test)[:, 1]

            for threshold in rf_thresholds:
                y_rf_threshold = (y_rf_proba >= threshold).astype(int)
                cm = confusion_matrix(y_test, y_rf_threshold)

                rf_results.append({
                    'class_weight': class_weight,
                    'max_depth': 'None' if max_depth is None else max_depth,
                    'min_samples_leaf': min_samples_leaf,
                    'threshold': threshold,
                    'accuracy': accuracy_score(y_test, y_rf_threshold),
                    'precision_fatal': precision_score(y_test, y_rf_threshold, zero_division=0),
                    'recall_fatal': recall_score(y_test, y_rf_threshold, zero_division=0),
                    'f1_fatal': f1_score(y_test, y_rf_threshold, zero_division=0),
                    'false_positives': cm[0, 1],
                    'false_negatives': cm[1, 0],
                    'true_positives': cm[1, 1]
                })

rf_results_df = pd.DataFrame(rf_results)
rf_results_df = rf_results_df.sort_values(by='f1_fatal', ascending=False)
print(rf_results_df.head(20))

best_rf = rf_results_df.iloc[0]
best_rf_label = (
    f"{best_rf['class_weight']} | depth {best_rf['max_depth']} | "
    f"leaf {best_rf['min_samples_leaf']}"
)

best_rf_thresholds = rf_results_df[
    (rf_results_df['class_weight'] == best_rf['class_weight']) &
    (rf_results_df['max_depth'] == best_rf['max_depth']) &
    (rf_results_df['min_samples_leaf'] == best_rf['min_samples_leaf'])
].sort_values(by='threshold')

plt.figure(figsize=(10, 6))
plt.plot(best_rf_thresholds['threshold'], best_rf_thresholds['precision_fatal'], marker='o', label='Precision fatal')
plt.plot(best_rf_thresholds['threshold'], best_rf_thresholds['recall_fatal'], marker='o', label='Recall fatal')
plt.plot(best_rf_thresholds['threshold'], best_rf_thresholds['f1_fatal'], marker='o', label='F1 fatal')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title(f'Random Forest - thresholds\n{best_rf_label}')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(fig_dir / 'random_forest_threshold_metrics.png')
plt.close()

best_by_config = (
    rf_results_df
    .sort_values(by='f1_fatal', ascending=False)
    .drop_duplicates(subset=['class_weight', 'max_depth', 'min_samples_leaf'])
    .head(10)
    .copy()
)

best_by_config['config'] = (
    best_by_config['class_weight'].astype(str) +
    '\ndepth=' + best_by_config['max_depth'].astype(str) +
    '\nleaf=' + best_by_config['min_samples_leaf'].astype(str) +
    '\nthr=' + best_by_config['threshold'].astype(str)
)

plt.figure(figsize=(14, 7))
x_pos = np.arange(len(best_by_config))
plt.bar(x_pos - 0.25, best_by_config['precision_fatal'], width=0.25, label='Precision fatal')
plt.bar(x_pos, best_by_config['recall_fatal'], width=0.25, label='Recall fatal')
plt.bar(x_pos + 0.25, best_by_config['f1_fatal'], width=0.25, label='F1 fatal')
plt.xticks(x_pos, best_by_config['config'], rotation=45, ha='right')
plt.ylabel('Score')
plt.title('Random Forest - melhores configuraÃ§Ãµes por F1 fatal')
plt.grid(True, axis='y', alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(fig_dir / 'random_forest_config_comparison.png')
plt.close()


Acurácia: 0.9839366186945777
[[27057    27]
 [  415    17]]
              precision    recall  f1-score   support

           0       0.98      1.00      0.99     27084
           1       0.39      0.04      0.07       432

    accuracy                           0.98     27516
   macro avg       0.69      0.52      0.53     27516
weighted avg       0.98      0.98      0.98     27516

           class_weight max_depth  min_samples_leaf  threshold  accuracy  precision_fatal  recall_fatal  f1_fatal  false_positives  false_negatives  true_positives
38             balanced      None                 1       0.20  0.981066         0.348123      0.236111  0.281379              191              330             102
92   balanced_subsample      None                 1       0.20  0.980411         0.318644      0.217593  0.258597              201              338              94
47             balanced      None                20       0.50  0.961404         0.183735      0.423611  0.256303              813              249             183
101  balanced_subsample      None                20       0.50  0.960205         0.179090      0.428241  0.252560              848              247             185
37             balanced      None                 1       0.10  0.966783         0.194937      0.356481  0.252046              636              278             154
91   balanced_subsample      None                 1       0.10  0.965983         0.184211      0.340278  0.239024              651              285             147
93   balanced_subsample      None                 1       0.30  0.983282         0.413580      0.155093  0.225589               95              365              67
39             balanced      None                 1       0.30  0.982846         0.378049      0.143519  0.208054              102              370              62
36             balanced      None                 1       0.05  0.927678         0.108936      0.502315  0.179043             1775              215             217
46             balanced      None                20       0.40  0.917975         0.104807      0.560185  0.176578             2067              190             242
90   balanced_subsample      None                 1       0.05  0.926261         0.105679      0.495370  0.174196             1811              218             214
100  balanced_subsample      None                20       0.40  0.918520         0.102373      0.539352  0.172083             2043              199             233
94   balanced_subsample      None                 1       0.40  0.983755         0.425743      0.099537  0.161351               58              389              43
107  balanced_subsample      None                50       0.50  0.906236         0.092564      0.564815  0.159061             2392              188             244
53             balanced      None                50       0.50  0.904129         0.089352      0.555556  0.153945             2446              192             240
40             balanced      None                 1       0.40  0.983464         0.383838      0.087963  0.143126               61              394              38
77   balanced_subsample        12                 1       0.50  0.909362         0.081914      0.467593  0.139406             2264              230             202
23             balanced        12                 1       0.50  0.906563         0.081081      0.479167  0.138693             2346              225             207
29             balanced        12                20       0.50  0.880869         0.077243      0.601852  0.136914             3106              172             260
83   balanced_subsample        12                20       0.50  0.880760         0.076672      0.597222  0.135897             3107              174             258
(venv) 
"""