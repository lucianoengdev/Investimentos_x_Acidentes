import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap

df = pd.read_csv('../data/processed/acidentes_tratados.csv')
"""
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
"""

bins_a = range(0, int(df['km_ajustado'].max()) + 5, 5)
df['km_bin'] = pd.cut(df['km_ajustado'], bins=bins_a, include_lowest=True)
df_heat = pd.crosstab(df['km_bin'], df['tipo_de_acidente']).sort_index()
plt.figure(figsize=(16,12))
sns.heatmap(df_heat, cmap='Reds', linewidths= 0.5, annot=False)
plt.title('Heatmap Tipo de acidente')
plt.xlabel('Tipo de Acidente')
plt.ylabel('km')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('../reports/figures/02_hotspots/heatmap_tipo_de_acidente.png')

df_fatal = df[df['acidente_fatal'] == 1]
df_heat_fatal = pd.crosstab(df_fatal['km_bin'], df_fatal['tipo_de_acidente']).sort_index()
plt.figure(figsize=(16,12))
sns.heatmap(df_heat_fatal, cmap='Reds', linewidths= 0.5, annot=False)
plt.title('Heatmap Acidentes Fatais')
plt.xlabel('Tipo de Acidente Fatal')
plt.ylabel('km')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('../reports/figures/02_hotspots/heatmap_tipo_de_acidente_fatais.png')

bins_20 = range(0, int(df['km_ajustado'].max()) + 20, 20)
df['km_bin20'] = pd.cut(df['km_ajustado'], bins=bins_20, include_lowest=True)
plt.figure(figsize=(12, 8))
lista_de_tipos = df.groupby('tipo_de_acidente')['mortos'].sum().sort_values(ascending=False).head(10).index.tolist()
df_filtrado = df[df['tipo_de_acidente'].isin(lista_de_tipos)]
heatmap_df = (df_filtrado
    .pivot_table(
        index='km_bin20',               
        columns='tipo_de_acidente',    
        values='mortos',               
        aggfunc='sum',                
        fill_value=0                  
    )
)
sns.heatmap(heatmap_df, cmap='Reds', linewidths= 0.5, annot=True)
labels_quebrados = [
    '\n'.join(textwrap.wrap(label, width=18))
    for label in heatmap_df.columns
]
plt.xticks(ticks=range(len(labels_quebrados)), labels=labels_quebrados, rotation=90)
plt.xlabel('Tipo de acidente')
plt.ylabel('KM')
plt.title('Mortos por tipo de acidente e KM')
plt.tight_layout()
plt.savefig('../reports/figures/02_hotspots/mais_mortes_tipo_de_acidente.png')


