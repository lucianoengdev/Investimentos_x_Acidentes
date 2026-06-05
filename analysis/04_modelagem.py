import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

df = pd.read_csv('../data/processed/acidentes_tratados.csv')

# Dada uma ocorrência de acidente, conseguimos prever se ela será fatal?

# Colunas a manter = [['horario', 'trecho', 'sentido', 'tipo_de_acidente', 'automovel', 'bicicleta',  'caminhao', 'moto', 'onibus', 'outros', 'tracao_animal', 'transporte_de_cargas_especiais', 'trator_maquinas', 'utilitarios', 'km_ajustado', 'ano', 'mes', 'acidente_fatal', 'km_bin']] 
# obs. vou largar o km_bin e criar outro porque ele vem como texto e quero utiliza-lo categoricamente
# Colunas a excluir = [['data', 'km', 'n_da_ocorrencia', 'ilesos', 'levemente_feridos', 'moderadamente_feridos', 'gravemente_feridos', 'mortos']]


col_trash = ['data', 'km', 'tipo_de_ocorrencia', 'n_da_ocorrencia', 'ilesos', 'levemente_feridos', 'moderadamente_feridos', 'gravemente_feridos', 'mortos', 'km_bin']
df = df.drop(col_trash, axis=1)
df['tipo_de_acidente'] = (df['tipo_de_acidente'].replace('', 'Não informado').fillna('Não informado'))
bin20 = range(0, int(df['km_ajustado'].max()) + 20, 20)
df['km_bin'] = pd.cut(df['km_ajustado'], bins=bin20, include_lowest=True)

df['horario'] = pd.to_datetime(df['horario'], format='%H:%M:%S')
minutos = (df['horario'].dt.hour * 60 + df['horario'].dt.minute)
df['hora_sin'] = np.sin(2 * np.pi * minutos / 1440)
df['hora_cos'] = np.cos(2 * np.pi * minutos / 1440)

df['hour'] = df['horario'].dt.hour
df['bin_hora'] = pd.cut(df['hour'], bins=range(0, 25), labels=False, right=False)

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
df.loc[df['tipo_de_acidente'].str.contains('atropelamento'), 'tipo_de_acidente'] = 'atropelamento'
df.loc[df['tipo_de_acidente'].str.contains('choque'), 'tipo_de_acidente'] = 'choque'
df.loc[df['tipo_de_acidente'].str.contains('colisao'), 'tipo_de_acidente'] = 'colisao'
manter = ['atropelamento', 'choque', 'colisao', 'tombamento', 'engavetamento', 'capotamento']
df.loc[~df['tipo_de_acidente'].isin(manter), 'tipo_de_acidente'] = 'outros'

encoded = encoder.fit_transform(df[['tipo_de_acidente']])
df_encoded = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['tipo_de_acidente']))
df = pd.concat([df, df_encoded], axis=1)
df = df.drop(columns = ['trecho', 'tipo_de_acidente'])

X = df.drop(columns = 'acidente_fatal')
y = df['acidente_fatal']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = 42)

dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(X_train, y_train)
y_pred_dummy = dummy.predict(X_test)
accuracy_test = accuracy_score(y_test, y_pred_dummy)

print("Acurácia Dummy:", accuracy_test)
print("Distribuição y_test:")
print(y_test.value_counts(normalize=True))

fig_dir = Path('../reports/figures/04_modelos')
fig_dir.mkdir(parents=True, exist_ok=True)

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
