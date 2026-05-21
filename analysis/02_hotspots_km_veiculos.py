import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap

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

df['sentido_ajustado'] = df['sentido'].replace({'Sul': 'Crescente', 'Norte': 'Decrescente'})
df_sentido = (df.pivot_table(index= 'km_bin20', columns= 'sentido_ajustado', values= 'mortos', aggfunc='sum', fill_value=0))
plt.figure(figsize=(12,6))
df_sentido.plot(kind='bar')
plt.xlabel('Faixa de KM')
plt.ylabel('Total de mortos')
plt.title('Mortos por faixa de KM e sentido')
plt.xticks(rotation= 45)
plt.legend(title='Sentido')
plt.tight_layout()
plt.savefig('../reports/figures/02_hotspots/mortes_por_sentido.png')

# cresc = df.loc[df['sentido_ajustado'] == 'Crescente', 'mortos'].sum() -> 1306
# decresc = df.loc[df['sentido_ajustado'] == 'Decrescente', 'mortos'].sum() -> 1145

col_veiculos = ['automovel', 'bicicleta', 'caminhao', 'moto', 'onibus', 'outros', 'tracao_animal', 'transporte_de_cargas_especiais', 'trator_maquinas', 'utilitarios']
df_fata = df[df['acidente_fatal'] == 1]
mortes_veiculos = df_fata[col_veiculos].sum().sort_values(ascending= False)
plt.figure(figsize=(12,6))
mortes_veiculos.plot(kind= 'bar')
labels_queb = ['\n'.join(textwrap.wrap(label, width=18))
               for label in mortes_veiculos.index]
plt.xticks(ticks=range(len(labels_queb)), labels=labels_queb, rotation= 45)
plt.xlabel('Veículos')
plt.ylabel('Quantidade envolvida em acidentes fatais')
plt.title('Veículos envolvidos em acidentes fatais')
plt.tight_layout()
plt.savefig('../reports/figures/02_hotspots/mortes_por_tipo_de_veiculos.png')
plt.close()

top_veiculos = mortes_veiculos.head(4).index.tolist()
df_fata_veiculos = df_fata[['ano', 'km_bin20', 'mortos'] + top_veiculos].copy()
df_veiculos_long = df_fata_veiculos.melt(
    id_vars=['ano', 'km_bin20', 'mortos'],
    value_vars=top_veiculos,
    var_name='veiculo',
    value_name='quantidade_veiculo'
)
df_veiculos_long = df_veiculos_long[df_veiculos_long['quantidade_veiculo'] > 0]
fig, axes = plt.subplots(2, 2, figsize=(18, 14), sharex=True, sharey=True)
for veiculo, ax in zip(top_veiculos, axes.flatten()):
    df_veiculo = df_veiculos_long[df_veiculos_long['veiculo'] == veiculo]
    matriz_veiculo = (
        df_veiculo
        .pivot_table(
            index='km_bin20',
            columns='ano',
            values='mortos',
            aggfunc='sum',
            fill_value=0
        )
        .sort_index()
    )
    sns.heatmap(
        matriz_veiculo,
        cmap='Reds',
        linewidths=0.3,
        annot=False,
        ax=ax,
        cbar=True
    )
    ax.set_title(f'Mortes envolvendo {veiculo}')
    ax.set_xlabel('Ano')
    ax.set_ylabel('Faixa de KM')
plt.tight_layout()
plt.savefig('../reports/figures/02_hotspots/mortes_por_km_ano_e_veiculo.png')

# tomb = df[df['tipo_de_acidente'] == 'Tombamento']
# tomb_sentido = tomb.groupby('sentido_ajustado')['mortos'].sum()
# Crescente      119
# Decrescente    111

veiculos_interesse = ['automovel', 'caminhao', 'moto']
df_veic_tipo = df[['tipo_de_acidente', 'mortos'] + veiculos_interesse].copy()
df_veic_tipo_long = df_veic_tipo.melt(
    id_vars=['tipo_de_acidente', 'mortos'],
    value_vars=veiculos_interesse,
    var_name='veiculo',
    value_name='quantidade_veiculo'
)
df_veic_tipo_long = df_veic_tipo_long[df_veic_tipo_long['quantidade_veiculo'] > 0]
top_tipos_veiculos = (
    df_veic_tipo_long
    .groupby('tipo_de_acidente')['mortos']
    .sum()
    .sort_values(ascending=False)
    .head(12)
    .index
    .tolist()
)
df_veic_tipo_long = df_veic_tipo_long[df_veic_tipo_long['tipo_de_acidente'].isin(top_tipos_veiculos)]
matriz_veiculo_tipo = (
    df_veic_tipo_long
    .pivot_table(
        index='veiculo',
        columns='tipo_de_acidente',
        values='mortos',
        aggfunc='sum',
        fill_value=0
    )
    .reindex(veiculos_interesse)
)
labels_tipos = [
    '\n'.join(textwrap.wrap(label, width=16))
    for label in matriz_veiculo_tipo.columns
]
plt.figure(figsize=(16, 5))
sns.heatmap(
    matriz_veiculo_tipo,
    cmap='Reds',
    linewidths=0.5,
    annot=True,
    fmt='.0f'
)
plt.xticks(ticks=range(len(labels_tipos)), labels=labels_tipos, rotation=45)
plt.xlabel('Tipo de acidente')
plt.ylabel('Veiculo envolvido')
plt.title('Mortes por tipo de acidente e veiculo envolvido')
plt.tight_layout()
plt.savefig('../reports/figures/02_hotspots/mortes_por_tipo_acidente_e_veiculo.png')


"""
- Interpretação -
A análise espacial por faixas de km indica que os pontos mais críticos se concentram principalmente no início e no fim do trecho ajustado da rodovia. As faixas próximas ao começo do trajeto e as faixas finais apresentam maior concentração de mortes e acidentes fatais. Uma hipótese possível é que essas regiões estejam associadas a áreas com maior presença humana, acessos urbanos ou travessias, mas essa interpretação exigiria validação com dados externos de uso do solo, população e infraestrutura viária.

A comparação entre volume total de acidentes e severidade mostra que quantidade de ocorrências não é, isoladamente, o melhor indicador de perigo. Alguns tipos de acidente aparecem com alta frequência, como colisões, mas nem sempre concentram a maior quantidade de mortes. Por outro lado, atropelamentos, especialmente envolvendo moradores, aparecem com frequência menor em relação às colisões, mas representam uma parcela muito relevante das mortes. Isso sugere que a análise de risco precisa considerar a gravidade do acidente, não apenas o número total de registros.

Nos gráficos por tipo de veículo, automóveis, caminhões e motos apresentam padrões espaciais semelhantes em algumas regiões críticas, com maior intensidade nas mesmas faixas de km. No entanto, a quantidade de registros envolvendo automóveis é maior, seguida por caminhões e motos. Essa concentração recorrente em determinadas faixas sugere a existência de zonas de perigo que merecem análise mais detalhada, independentemente do tipo de veículo envolvido.

A análise por sentido da via indica que, na maior parte do trecho, os padrões entre os sentidos crescente e decrescente são relativamente parecidos. Ainda assim, algumas faixas apresentam diferenças relevantes. Nos intervalos aproximados de 0 a 20 km, 40 a 60 km, 80 a 100 km, 360 a 380 km e 540 a 560 km, o sentido crescente apresenta valores bem superiores ao decrescente. Já nos intervalos de 60 a 80 km e 140 a 160 km ocorre o contrário. Isso sugere que o sentido da via pode influenciar o risco em pontos específicos, embora a tendência geral do trecho seja de certa similaridade entre os sentidos.

De forma geral, as regiões com maior concentração de acidentes graves e mortes aparecem principalmente nas faixas iniciais e finais do trecho ajustado, especialmente em torno de 0 a 40 km, 520 a 540 km e acima de 550 km. Além disso, os atropelamentos se destacam como um dos principais tipos de ocorrência associados à mortalidade, reforçando a necessidade de investigar melhor características locais dessas regiões, como travessias, ocupação urbana, acessos laterais e presença de pedestres.


Até aqui, os dados processados mostram:

Total de registros: 137.576 acidentes.
Acidentes fatais: 2.161 ocorrências.
Mortes registradas: 2.451 mortes.
Sentido crescente: 1.306 mortes.
Sentido decrescente: 1.145 mortes.
Tombamentos por sentido: 119 mortes no crescente e 111 no decrescente, diferença pequena.
Faixas de km com mais mortes:

0 a 20 km: 302 mortes.
540 a 560 km: 226 mortes.
40 a 60 km: 142 mortes.
500 a 520 km: 135 mortes.
520 a 540 km: 133 mortes.
20 a 40 km: 129 mortes.
Tipos de acidente com mais mortes:

Atropelamento - Morador: 321 mortes.
Outros - Sequência: 287 mortes.
Colisão Traseira: 242 mortes.
Tombamento: 230 mortes.
Capotamento: 226 mortes.
Atropelamento - Andarilho: 182 mortes.
Atropelamento de pedestre: 123 mortes.
Veículos mais presentes em acidentes fatais:

Automóvel: 1.199 envolvimentos.
Caminhão: 1.144 envolvimentos.
Moto: 416 envolvimentos.
Ônibus: 103 envolvimentos.
No novo cruzamento veículo x tipo de acidente, alguns sinais interessantes:

Em Atropelamento - Morador, automóvel aparece com 181 mortes, caminhão com 41, moto com 7.
Em Colisão Traseira, caminhão aparece com 184 mortes, automóvel com 64, moto com 77.
Em Tombamento, caminhão aparece com 176 mortes, moto com 24, automóvel com 20.
Em Capotamento, automóvel aparece com 158 mortes, caminhão com 47, moto com 0.
"""