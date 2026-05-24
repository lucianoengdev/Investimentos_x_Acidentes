import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

df = pd.read_csv('../data/processed/investimentos_tratados.csv')

df = df[df['concessionaria'] == 'AUTOPISTA FERNÃO DIAS']
df.columns = df.columns.str.lower()
df['ano'] = df['ano'].astype(int)
df['valor'] = pd.to_numeric(df['valor'])

df_ipca = pd.DataFrame({
    'ano': [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
    'ipca_anual': [5.91, 6.50, 5.84, 5.91, 6.41, 10.67, 6.29, 2.95, 3.75, 4.31, 4.52, 10.06, 5.79, 4.62, 4.83],
    'fator_anual': [1.0591, 1.0650, 1.0584, 1.0591, 1.0641, 1.1067, 1.0629, 1.0295, 1.0375, 1.0431, 1.0452, 1.1006, 1.0579, 1.0462, 1.0483]
})
"""
Esses valores foram obtidos de:
https://sidra.ibge.gov.br/tabela/1737
Para ver no site visual, entre na Tabela 1737 e selecione:
variável: IPCA - Variação acumulada no ano;
período: dezembro de cada ano;
território: Brasil.

These values were obtained from:
https://sidra.ibge.gov.br/tabela/1737

To view them on the website, open Table 1737 and select:
- variable: IPCA - Accumulated annual variation
- period: December of each year
- territory: Brazil

2010 não calcula inflação pois é o ano base*
"""
df = pd.merge(df, df_ipca, on='ano', how='inner')
df = df.sort_values('ano', ascending=False)
df['fator_para_2024'] = df['fator_anual'].cumprod().shift(1, fill_value=1)
df = df.sort_values('ano')
df['valor_corrigido'] = df['valor'] * df['fator_para_2024']
df = df.sort_values('ano', ascending=True)

plt.figure(figsize=(12,6))
plt.plot(df['ano'], df['valor_corrigido'], color='green', marker='o', linestyle='-', label='Investimento Corrigido')
plt.plot(df['ano'], df['valor'], color='blue', marker='o', linestyle='-', label='Investimento')
ax = plt.gca()
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'R$ {x/1_000_000:.0f} Mi'))
plt.title('Investimento ao longo dos anos')
plt.xlabel('Ano')
plt.xticks(df['ano'], rotation=0)
plt.tight_layout()
plt.savefig('../reports/figures/03_investimentos/investimento_por_ano.png')

df_a = pd.read_csv('../data/processed/acidentes_tratados.csv')
df_aci_ano = df_a.groupby('ano')['acidente_fatal'].sum().reset_index()

df_aux0 = pd.merge(df, df_aci_ano, on='ano', how='outer')
df_aux0.set_index('ano', inplace=True)
df_aux0.drop(2026, inplace=True)

plt.figure(figsize=(12,6))
ax0 = df_aux0['valor_corrigido'].plot(kind='line', color='blue', marker='o', linestyle='-', label='Investimentos')
ax0.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'R$ {x/1_000_000:.0f} Mi'))
df_aux0['acidente_fatal'].plot(kind='line', secondary_y=True, color='red', marker='x', linestyle='-', ax=ax0, label='Acidentes Fatais')
ax0.set_ylabel('Investimentos')
ax0.right_ax.set_ylabel('Acidentes Fatais')
plt.title('Acidentes Fatais x Investimentos')
plt.savefig('../reports/figures/03_investimentos/acidentesfatais_x_investimento.png')

plt.figure(figsize=(12,6))
df_aux0['investimento_ano_anterior'] = df_aux0['valor_corrigido'].shift(1)
df_aux0.plot.scatter(x='investimento_ano_anterior', y='acidente_fatal', color='purple')
for ano, x, y in zip(df_aux0.index, df_aux0['investimento_ano_anterior'], df_aux0['acidente_fatal']):
    plt.text(x, y, str(ano))
plt.xlabel('Investimento do Ano Anterior')
plt.ylabel('Acidentes Fatais')
plt.title('Acidentes Fatais x Investimento ano anterior')
plt.savefig('../reports/figures/03_investimentos/acidentesfatais_x_investimento_anoanterior.png')

df_aux0['soma_movel3'] = df_aux0['valor_corrigido'].rolling(window=3).sum()
plt.figure(figsize=(12,6))
plt.plot(df_aux0.index, df_aux0['soma_movel3'], color='blue', marker='x', linestyle='-')
plt.ylabel('Valor')
plt.xlabel('Ano')
plt.xticks(df_aux0.index, rotation=45)
plt.title('Investimento somado dos últimos 3 anos')
ax3 = plt.gca()
ax3.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'R$ {x/1_000_000:.0f} Mi'))
plt.savefig('../reports/figures/03_investimentos/investimento_somado_3_ultimos_anos.png')

df_aux0['investimento_medio'] = df_aux0['valor_corrigido'].expanding().mean()
plt.figure(figsize=(12,6))
plt.bar(df_aux0.index, df_aux0['investimento_medio'], color='blue')
plt.xlabel('Ano')
plt.ylabel('Valor')
plt.title('Valor da média de investimentos até o ano')
ax2 = plt.gca()
ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'R$ {x/1_000_000:.0f} Mi'))
plt.ylim(0, 600000000)
plt.xticks(df_aux0.index, rotation=45)
plt.tight_layout()
plt.savefig('../reports/figures/03_investimentos/media_valores_ano.png')


def classificar_fase(ano):
    if ano <= 2014:
        return '2010-2014 estrutural'
    if ano <= 2020:
        return '2015-2020 madura'
    return '2021-2025 final'

df_aux0['fase'] = [classificar_fase(ano) for ano in df_aux0.index]
ordem_fases = ['2010-2014 estrutural', '2015-2020 madura', '2021-2025 final']

fig, ax4b = plt.subplots(figsize=(12,6))
ax4b.plot(df_aux0.index, df_aux0['valor_corrigido'], color='green', marker='o', linestyle='-', label='Investimento Corrigido')
ax4b.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'R$ {x/1_000_000:.0f} Mi'))
ax4b.set_ylabel('Investimento Corrigido')
ax4b.set_xlabel('Ano')
ax4b.set_xticks(df_aux0.index)
ax4b.tick_params(axis='x', rotation=45)
ax4b.axvspan(2010, 2014, color='green', alpha=0.08, label='Fase estrutural')
ax4b.axvspan(2015, 2020, color='blue', alpha=0.06, label='Fase madura')
ax4b.axvspan(2021, 2025, color='orange', alpha=0.08, label='Fase final')

ax4b_2 = ax4b.twinx()
ax4b_2.plot(df_aux0.index, df_aux0['acidente_fatal'], color='red', marker='x', linestyle='-', label='Acidentes Fatais')
ax4b_2.set_ylabel('Acidentes Fatais')

linhas_1, labels_1 = ax4b.get_legend_handles_labels()
linhas_2, labels_2 = ax4b_2.get_legend_handles_labels()
ax4b.legend(linhas_1 + linhas_2, labels_1 + labels_2, loc='upper right')
plt.title('Fase 4.b - Investimento corrigido x acidentes fatais por fase')
plt.tight_layout()
plt.savefig('../reports/figures/03_investimentos/4b_investimento_acidentes_por_fase.png')
plt.close()

resumo_fases = (
    df_aux0
    .groupby('fase')
    .agg(
        media_acidentes_fatais=('acidente_fatal', 'mean'),
        media_investimento_corrigido=('valor_corrigido', 'mean')
    )
    .reindex(ordem_fases)
)

fig, axes = plt.subplots(1, 2, figsize=(14,6))
axes[0].bar(resumo_fases.index, resumo_fases['media_acidentes_fatais'], color='red')
axes[0].set_title('Media de acidentes fatais por fase')
axes[0].set_ylabel('Acidentes Fatais')
axes[0].tick_params(axis='x', rotation=30)

axes[1].bar(resumo_fases.index, resumo_fases['media_investimento_corrigido'], color='green')
axes[1].set_title('Media de investimento corrigido por fase')
axes[1].set_ylabel('Investimento Corrigido')
axes[1].tick_params(axis='x', rotation=30)
axes[1].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'R$ {x/1_000_000:.0f} Mi'))

plt.tight_layout()
plt.savefig('../reports/figures/03_investimentos/4b_medias_por_fase.png')
plt.close()

cores_fases = {
    '2010-2014 estrutural': 'green',
    '2015-2020 madura': 'blue',
    '2021-2025 final': 'orange'
}

fig, ax_lag = plt.subplots(figsize=(12,6))
for fase in ordem_fases:
    dados_fase = df_aux0[df_aux0['fase'] == fase].dropna(subset=['investimento_ano_anterior', 'acidente_fatal'])
    ax_lag.scatter(
        dados_fase['investimento_ano_anterior'],
        dados_fase['acidente_fatal'],
        color=cores_fases[fase],
        label=fase
    )
    for ano, x, y in zip(dados_fase.index, dados_fase['investimento_ano_anterior'], dados_fase['acidente_fatal']):
        ax_lag.text(x, y, str(ano))

ax_lag.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'R$ {x/1_000_000:.0f} Mi'))
ax_lag.set_xlabel('Investimento corrigido do ano anterior')
ax_lag.set_ylabel('Acidentes Fatais')
ax_lag.set_title('Fase 4.b - Lag 1 por fase')
ax_lag.legend(title='Fase')
plt.tight_layout()
plt.savefig('../reports/figures/03_investimentos/4b_lag1_por_fase.png')
plt.close()

fig, ax_roll = plt.subplots(figsize=(12,6))
ax_roll.plot(df_aux0.index, df_aux0['soma_movel3'], color='green', marker='o', linestyle='-', label='Investimento acumulado 3 anos')
ax_roll.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'R$ {x/1_000_000:.0f} Mi'))
ax_roll.set_ylabel('Investimento acumulado 3 anos')
ax_roll.set_xlabel('Ano')
ax_roll.set_xticks(df_aux0.index)
ax_roll.tick_params(axis='x', rotation=45)
ax_roll.axvspan(2010, 2014, color='green', alpha=0.08)
ax_roll.axvspan(2015, 2020, color='blue', alpha=0.06)
ax_roll.axvspan(2021, 2025, color='orange', alpha=0.08)

ax_roll2 = ax_roll.twinx()
ax_roll2.plot(df_aux0.index, df_aux0['acidente_fatal'], color='red', marker='x', linestyle='-', label='Acidentes Fatais')
ax_roll2.set_ylabel('Acidentes Fatais')

linhas_roll_1, labels_roll_1 = ax_roll.get_legend_handles_labels()
linhas_roll_2, labels_roll_2 = ax_roll2.get_legend_handles_labels()
ax_roll.legend(linhas_roll_1 + linhas_roll_2, labels_roll_1 + labels_roll_2, loc='upper right')
plt.title('Fase 4.b - Investimento acumulado 3 anos x acidentes fatais')
plt.tight_layout()
plt.savefig('../reports/figures/03_investimentos/4b_investimento_3anos_acidentes_por_fase.png')
plt.close()
