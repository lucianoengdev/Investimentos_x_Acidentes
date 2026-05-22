import pandas as pd

df = pd.read_csv('../data/processed/investimentos_tratados.csv')

df = df[df['concessionaria'] == 'AUTOPISTA FERNÃO DIAS']
df.columns = df.columns.str.lower()
df['ano'] = df['ano'].astype(int)
df['valor'] = pd.to_numeric(df['valor'])

df_ipca = pd.DataFrame({
    'ano': [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
    'ipca_anual': [0, 6.50, 5.84, 5.91, 6.41, 10.67, 6.29, 2.95, 3.75, 4.31, 4.52, 10.06, 5.79, 4.62, 4.83],
    'fator_anual': [1.00, 1.0650, 1.0584, 1.0591, 1.0641, 1.1067, 1.0629, 1.0295, 1.0375, 1.0431, 1.0452, 1.1006, 1.0579, 1.0462, 1.0483]
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
df['fator_para_ano'] = df['fator_anual'].cumprod()
df['valor_corrigido'] = df['valor'] * df['fator_para_ano']

print(df)