import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('../data/processed/acidentes_tratados.csv')

cols_ano = ['acidente_fatal', 'mortos']
df_fatal = df.groupby('ano')[cols_ano].sum().reset_index()
df_fatal['acidentes'] = df.groupby('ano').size().values
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

plt.figure(figsize=(12,6))
plt.plot(df_fatal['ano'], df_fatal['acidentes'], color='blue', marker='o', linestyle='-')
plt.title('Acidentes por ano')
plt.xlabel('Ano')
plt.ylabel('Acidentes')
plt.xticks(df_fatal['ano'], rotation=45)
plt.grid(True, alpha= 0.2)
plt.savefig('../reports/figures/01_initial_analysis/acidentes_por_ano.png')

anos = range(2010, 2026)
fig, axes = plt.subplots(4, 4, figsize=(22, 16))
for ano, ax in zip(anos, axes.flatten()):
    df_ano = df[df['ano'] == ano]
    tipo_de_acidentes = df_ano['tipo_de_acidente'].value_counts().head(10)
    ax.barh(tipo_de_acidentes.index, tipo_de_acidentes.values)
    ax.set_title(str(ano))
    ax.set_xlabel('Quantidade')
    ax.invert_yaxis()
plt.tight_layout()
plt.savefig('../reports/figures/01_initial_analysis/tipo_de_acidentes_por_ano.png')

plt.figure(figsize=(12,6))
plt.hist(df['km_ajustado'].dropna(), bins=20)
plt.title('Histograma de acidentes por KM')
plt.xlabel('KM')
plt.ylabel('Frequência')
plt.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig('../reports/figures/01_initial_analysis/histograma_acidentes_km.png')
