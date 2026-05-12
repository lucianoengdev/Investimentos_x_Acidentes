import pandas as pd
import numpy as np

df_raw_a = pd.read_csv('../data/raw/acidentes.csv', sep=';', decimal=',')
df_raw_i = pd.read_csv('../data/raw/investimentos.csv', sep=';')

df_processed_a = df_raw_a.copy()
df_processed_i = df_raw_i.copy()

print(df_raw_a.describe())
print(df_raw_a.head())

print(df_raw_i.describe())
print(df_raw_i.head())


df_processed_a.to_csv('../data/processed/acidentes_tratados.csv', index=False)
df_processed_i.to_csv('../data/processed/investimentos_tratados.csv', index=False)