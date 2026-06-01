import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

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

print(df.columns)
print(df.head())
print(df.describe())


X = df.drop(columns = 'acidente_fatal')
y = df['acidente_fatal']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = 42)
