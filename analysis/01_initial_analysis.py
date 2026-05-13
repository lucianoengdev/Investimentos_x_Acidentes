import pandas as pd
import numpy as np

df_raw_a = pd.read_csv('../data/raw/acidentes.csv', sep=';', decimal=',')
df_raw_i = pd.read_csv('../data/raw/investimentos.csv', sep=';')

df_processed_a = df_raw_a.copy()
df_processed_i = df_raw_i.copy()

df_processed_a['data'] = pd.to_datetime(df_processed_a['data'], dayfirst=True)
df_processed_a['km'] = pd.to_numeric(df_processed_a['km'])
cols_to_int = ['automovel', 'bicicleta', 'caminhao', 'moto', 'onibus', 'outros', 'tracao_animal', 'transporte_de_cargas_especiais', 'trator_maquinas', 'utilitarios', 'ilesos', 'levemente_feridos', 'moderadamente_feridos', 'gravemente_feridos', 'mortos']
df_processed_a[cols_to_int] =df_processed_a[cols_to_int].apply(pd.to_numeric)

df_processed_a['ano'] = pd.to_datetime(df_processed_a['data'], dayfirst=True).dt.year
df_processed_a['mes'] = pd.to_datetime(df_processed_a['data'], dayfirst=True).dt.month
df_processed_a['acidente_fatal'] = np.where(df_processed_a['mortos'] > 0, 1, 0)

bins_a = range(0, int(df_processed_a['km'].max()) + 5, 5)
df_processed_a['km_bin'] = pd.cut(df_processed_a['km'], bins=bins_a)

"""
print("Number of NaN: ", df_processed_a.isna().sum())
Number of NaN:  
data                                0
horario                             0
n_da_ocorrencia                     0
tipo_de_ocorrencia                  0
km                                  0
trecho                              0
sentido                             0
tipo_de_acidente                  225
automovel                           0
bicicleta                           0
caminhao                            0
moto                                0
onibus                              0
outros                              0
tracao_animal                       0
transporte_de_cargas_especiais      0
trator_maquinas                     0
utilitarios                         0
ilesos                              0
levemente_feridos                   0
moderadamente_feridos               0
gravemente_feridos                  0
mortos                              0
ano                                 0
mes                                 0
acidente_fatal                      0
km_bin                             64
dtype: int64

print("Number of NaN: ", df_processed_i.isna().sum())
Number of NaN:  
concessionaria     0
ano                0
Valor             58
dtype: int64

print(df_processed_a['ano'].min(), df_processed_a['ano'].max())
2010 2026

print(sorted(df_processed_a['ano'].unique()))
[np.int32(2010), np.int32(2011), np.int32(2012), np.int32(2013), np.int32(2014), np.int32(2015), np.int32(2016), np.int32(2017), np.int32(2018), np.int32(2019), np.int32(2020), np.int32(2021), np.int32(2022), np.int32(2023), np.int32(2024), np.int32(2025), np.int32(2026)]

print("duplicates: ", df_processed_a.duplicated().sum)
duplicates:  <bound method Series.sum of 
0         False
1         False
2         False
3         False
4         False
          ...  
137571    False
137572    False
137573    False
137574    False
137575    False
Length: 137576, dtype: bool>

print("duplicates: ", df_processed_i.duplicated().sum)
duplicates:  <bound method Series.sum of 
0      False
1      False
2      False
3      False
4      False
       ...  
355    False
356    False
357    False
358    False
359    False
Length: 360, dtype: bool>

print("acidents with death: ", df_processed_a['acidente_fatal'].value_counts())
acidents with death:  acidente_fatal
0    135415 -> No Deaths
1      2161 -> Deaths

print(df_raw_a.describe())
Name: count, dtype: int64
        n_da_ocorrencia             km      automovel      bicicleta       caminhao  ...         ilesos  levemente_feridos  moderadamente_feridos  gravemente_feridos         mortos
count    137576.000000  137576.000000  137576.000000  137576.000000  137576.000000  ...  137576.000000      137576.000000          137576.000000       137576.000000  137576.000000
mean        240.229975     522.619949       0.872550       0.003693       0.309669  ...       1.572200           0.348709               0.058818            0.017816       0.017816
std         147.639026     300.757640       0.816695       0.063350       0.580012  ...       2.042658           0.800486               0.301044            0.147653       0.154486
min           1.000000       0.000000       0.000000       0.000000       0.000000  ...       0.000000           0.000000               0.000000            0.000000       0.000000
25%         111.000000     478.300000       0.000000       0.000000       0.000000  ...       1.000000           0.000000               0.000000            0.000000       0.000000
50%         233.000000     531.106000       1.000000       0.000000       0.000000  ...       1.000000           0.000000               0.000000            0.000000       0.000000
75%         348.000000     761.000000       1.000000       0.000000       1.000000  ...       2.000000           0.000000               0.000000            0.000000       0.000000
max        1000.000000     949.900000      28.000000       4.000000       9.000000  ...     116.000000          49.000000              33.000000            6.000000       8.000000
[8 rows x 17 columns]

print(df_raw_a.head())
    data   horario  n_da_ocorrencia tipo_de_ocorrencia     km     trecho sentido  ... trator_maquinas  utilitarios  ilesos  levemente_feridos  moderadamente_feridos  gravemente_feridos  mortos
0  01/01/2010  03:02:00               20         sem vítima  506.5  BR-381/MG     Sul  ...               0            0       1                  0                      0                   0       0
1  01/01/2010  06:16:00               39         sem vítima  767.0  BR-381/MG     Sul  ...               0            0       2                  0                      0                   0       0
2  01/01/2010  06:37:00               42         com vítima  567.0  BR-381/MG     Sul  ...               0            0       0                  1                      0                   0       0
3  01/01/2010  06:49:00               44         sem vítima  492.0  BR-381/MG   Norte  ...               0            0       1                  0                      0                   0       0
4  01/01/2010  07:11:00               48         sem vítima   76.5  BR-381/SP     Sul  ...               0            0       1                  0                      0                   0       0
[5 rows x 23 columns]

print(df_raw_i.describe())
               ano         Valor
count   360.000000  3.020000e+02
mean   2017.227778  2.014393e+08
std       4.380675  2.691211e+08
min    2010.000000  2.000000e+03
25%    2013.000000  3.924175e+07
50%    2017.000000  1.168285e+08
75%    2021.000000  2.439052e+08
max    2024.000000  2.054368e+09

print(df_raw_i.head())
  concessionaria   ano        Valor
0        CONCEPA  2010   36525000.0
1      NOVADUTRA  2010  220348000.0
2         CONCER  2010   13064000.0
3            CRT  2010   34092000.0
4         ECOSUL  2010   19100000.0

print(df_processed_a.shape, df_processed_i.shape)
(137576, 27) (360, 3)
"""

df_processed_a.to_csv('../data/processed/acidentes_tratados.csv', index=False)
df_processed_i.to_csv('../data/processed/investimentos_tratados.csv', index=False)