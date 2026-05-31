import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv('../data/processed/acidentes_tratados.csv')

# Dada uma ocorrência de acidente, conseguimos prever se ela será fatal?

# Colunas a manter = [['horario', 'trecho', 'sentido', 'tipo_de_acidente', 'automovel', 'bicicleta',  'caminhao', 'moto', 'onibus', 'outros', 'tracao_animal', 'transporte_de_cargas_especiais', 'trator_maquinas', 'utilitarios', 'km_ajustado', 'ano', 'mes', 'acidente_fatal', 'km_bin']]
# Colunas a excluir = [['data', 'km', 'n_da_ocorrencia', 'ilesos', 'levemente_feridos', 'moderadamente_feridos', 'gravemente_feridos', 'mortos']]

col_trash = ['data', 'km', 'tipo_de_ocorrencia', 'n_da_ocorrencia', 'ilesos', 'levemente_feridos', 'moderadamente_feridos', 'gravemente_feridos', 'mortos']
df = df.drop(col_trash, axis=1)
nan_count = df['tipo_de_acidente'].isna().sum()

X = df.drop(columns = 'acidente_fatal')
y = df['acidente_fatal']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = 42)
