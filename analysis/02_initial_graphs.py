import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('../data/processed/acidentes_tratados.csv')

cols_ano = ['acidente_fatal', 'mortos']
df_fatal = df.groupby('ano')[cols_ano].sum().reset_index()
df_fatal = df_fatal[df_fatal['ano'] < 2026]
plt.figure(figsize=(12, 6))
plt.plot(df_fatal['ano'], df_fatal['acidente_fatal'], color='blue', marker='o', linestyle='-')
plt.title('Acidentes fatais por ano')
plt.xlabel('Ano')
plt.ylabel('Quantidade de acidentes fatais')
plt.xticks(df_fatal['ano'], rotation = 45)
plt.grid(True, alpha= 0.3)

plt.savefig('../reports/figures/01_initial_analysis/acidentes_fatais_por_ano.png')

plt.figure(figsize=(12, 6))
plt.plot(df_fatal['ano'], df_fatal['mortos'], color='blue', marker='o', linestyle='-')
plt.title('Mortos por ano')
plt.xlabel('Ano')
plt.ylabel('Mortos')
plt.xticks(df_fatal['ano'], rotation=45)
plt.grid(True, alpha= 0.2)

plt.savefig('../reports/figures/01_initial_analysis/mortos_por_ano.png')
