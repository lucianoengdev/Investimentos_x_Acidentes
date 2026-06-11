# Modeling Baseline and Next Steps 

This note documents the first machine learning baseline results and the next modeling direction after the exploratory analysis.

### Modeling Objective

The modeling question is:

```text
Given the characteristics of a crash occurrence, can a model predict whether the crash will be fatal?
```

The target variable is:

```text
acidente_fatal
```

This is a binary classification problem:

```text
0 = non-fatal crash
1 = fatal crash
```

The modeling task was intentionally moved to occurrence-level crash data because the annual investment dataset has too few observations for robust predictive modeling. Investment analysis remains useful as exploratory context, while machine learning is better suited to the crash-level dataset.

### Dataset Strategy

The next modeling iteration should start again from:

```text
data/processed/acidentes_tratados.csv
```

instead of saving and reloading the already-modeled dataframe from `04_modelagem.py`.

The reason is methodological control. The current modeling dataframe already contains specific decisions about feature engineering, category grouping, normalization, encoding, and column removal. If that dataframe is saved and reused directly, it becomes harder to determine whether a future improvement came from a real feature change or from a leftover transformation.

Starting from the processed crash dataset allows each modeling experiment to define its own feature set clearly:

* which columns are removed;
* how `tipo_de_acidente` is grouped;
* whether `trecho` is included or removed;
* how `km_ajustado` and `km_bin` are represented;
* how time-of-day features are created;
* which categorical variables are encoded;
* which numeric variables are scaled.

The processed dataset is the correct starting point because it already contains the core cleaning decisions from the data preparation phase, including corrected kilometer logic and the `acidente_fatal` target. Model-specific transformations should remain inside the modeling script or reusable modeling functions.

### Baseline Results

The first baseline was a `DummyClassifier` using `most_frequent`.

Result:

```text
accuracy = 0.9843
```

The test distribution was:

```text
0 = 98.43%
1 = 1.57%
```

This result showed that accuracy is misleading for this problem. A model can achieve very high accuracy by predicting almost every crash as non-fatal. Therefore, model evaluation must focus on the fatal class.

Important metrics:

* precision for class `1`;
* recall for class `1`;
* F1-score for class `1`;
* confusion matrix;
* false positives;
* false negatives;
* true positives.

### Logistic Regression Baseline

The first real model was Logistic Regression with:

```text
class_weight = balanced
```

After scaling numeric variables, the convergence warning disappeared, but the metrics changed only slightly. This was expected: scaling helps optimization, but does not necessarily solve class imbalance.

At threshold `0.5`, the model produced:

```text
accuracy = 0.8008
precision fatal = 0.06
recall fatal = 0.74
F1 fatal = 0.10
```

Confusion matrix:

```text
[[21713  5371]
 [  111   321]]
```

Interpretation:

* the model found 321 of 432 fatal crashes;
* recall was high compared with the dummy baseline;
* precision was very low;
* the model produced many false positives.

Threshold testing showed that higher thresholds reduced false positives and improved the fatal-class F1-score.

After separating pedestrian-related crash categories more carefully, the Logistic Regression result at threshold `0.8` was:

```text
accuracy = 0.9614
precision fatal = 0.18
recall fatal = 0.41
F1 fatal = 0.25
```

Confusion matrix:

```text
[[26277   807]
 [  255   177]]
```

This became the strongest Logistic Regression configuration so far. It is more selective than lower thresholds while still capturing a meaningful share of fatal crashes.

### Decision Tree Baseline

A first `DecisionTreeClassifier` with:

```text
class_weight = balanced
```

produced:

```text
accuracy = 0.9750
precision fatal = 0.17
recall fatal = 0.15
F1 fatal = 0.15
```

Confusion matrix:

```text
[[26766   318]
 [  369    63]]
```

The tree was more conservative than Logistic Regression. It generated fewer false positives but missed most fatal crashes.

Additional tests were run with:

```text
max_depth = 4, 6, 8, 10
min_samples_leaf = 20, 50, 100
```

The best tested tree configuration reached approximately:

```text
test F1 fatal = 0.110
```

This was still weaker than Logistic Regression with threshold adjustment.

### Random Forest Baseline

The first `RandomForestClassifier` with:

```text
class_weight = balanced
```

and default threshold `0.5` produced:

```text
accuracy = 0.9839
precision fatal = 0.39
recall fatal = 0.04
F1 fatal = 0.07
```

Confusion matrix:

```text
[[27057    27]
 [  415    17]]
```

This model was extremely conservative and rarely predicted the fatal class.

After testing thresholds and hyperparameters, the best Random Forest configuration by fatal-class F1 was:

```text
class_weight = balanced
max_depth = None
min_samples_leaf = 1
threshold = 0.20
```

Result:

```text
accuracy = 0.9811
precision fatal = 0.3481
recall fatal = 0.2361
F1 fatal = 0.2814
false positives = 191
false negatives = 330
true positives = 102
```

This became the strongest model so far by fatal-class F1-score. Compared with Logistic Regression, it is more precise and produces fewer false positives, but it captures fewer fatal crashes.

### Current Model Interpretation

The current models show different trade-offs:

* DummyClassifier proves that accuracy alone is not enough.
* Logistic Regression captures more fatal crashes but generates many false positives.
* Decision Tree is interpretable but weak for the fatal class.
* Random Forest currently achieves the best fatal-class F1-score but has lower recall than Logistic Regression.

This suggests that the next modeling step should focus not only on algorithm tuning but also on improving feature representation.

### Next Feature Engineering Direction

The next modeling iteration will test whether more careful feature engineering improves fatal-crash prediction.

Planned tests:

#### 1. Remove `trecho`

Rationale: `km_ajustado` and `km_bin_id` already represent road position more precisely. `trecho` may be redundant and add little predictive information.

#### 2. Separate collision categories

Current grouping may be too broad. Collision types should be separated into more meaningful categories such as:

```text
colisao_traseira
colisao_frontal
colisao_lateral
colisao_transversal
colisao_outros
```

Rationale: rear-end and frontal collisions have different severity patterns. Aggregating all collision types may hide important fatal-risk signals.

#### 3. Test better spatial features

Possible alternatives:

* keep `km_ajustado`;
* keep or remove `km_bin_id`;
* use one-hot encoding for kilometer bins;
* create hotspot indicators such as `zona_inicio`, `zona_final`, or `zona_hotspot`.

Rationale: exploratory analysis showed a strong spatial concentration of fatal crashes, especially at the beginning and end of the corrected road sequence.

#### 4. Simplify time features

Current time features include:

```text
hora_sin
hora_cos
hour
bin_hora
```

These may be redundant.

Future tests should compare:

* cyclical representation only (`hora_sin`, `hora_cos`);
* binned representation only (`bin_hora`);
* period-of-day categories.

#### 5. Add vehicle interaction features

Possible features:

* `total_veiculos`;
* `envolve_caminhao`;
* `envolve_moto`;
* interactions between crash type and vehicle involvement.

Rationale: exploratory analysis showed that vehicle involvement varies considerably by crash type, especially for trucks, motorcycles, and pedestrian-related crashes.

### Experiment Discipline

Future tests should modify one feature-engineering decision at a time.

Recommended process:

```text
1. Define baseline feature set.
2. Train the selected models with fixed thresholds.
3. Save metrics.
4. Change one feature decision.
5. Train again.
6. Compare fatal-class precision, recall and F1.
```

The main comparison table should contain:

```text
model
feature_version
threshold
precision_fatal
recall_fatal
f1_fatal
false_positives
false_negatives
true_positives
```

The next practical step is to create a cleaner modeling experiment pipeline that starts from `acidentes_tratados.csv`, rebuilds the feature set, and compares the reference models under controlled feature-engineering changes.

---


# Linha de Base da Modelagem e Próximos Passos - PT-BR

Esta nota documenta os primeiros resultados de linha de base (baseline) de aprendizado de máquina e a próxima direção da modelagem após a análise exploratória.

### Objetivo da Modelagem

A pergunta de modelagem é:

```text
Dadas as características de uma ocorrência de acidente, um modelo consegue prever se o acidente será fatal?
```

A variável alvo é:

```text
acidente_fatal
```

Este é um problema de classificação binária:

```text
0 = acidente não fatal
1 = acidente fatal
```

A etapa de modelagem foi intencionalmente transferida para os dados de ocorrências de acidentes porque o conjunto de dados anual de investimentos possui poucas observações para uma modelagem preditiva robusta. A análise de investimentos continua sendo útil como contexto exploratório, enquanto técnicas de aprendizado de máquina são mais adequadas ao conjunto de dados em nível de ocorrência.

### Estratégia do Dataset

A próxima iteração de modelagem deve começar novamente a partir de:

```text
data/processed/acidentes_tratados.csv
```

em vez de salvar e reutilizar diretamente o dataframe já transformado em `04_modelagem.py`.

O motivo é controle metodológico. O dataframe atual já incorpora decisões específicas de engenharia de atributos, agrupamento de categorias, normalização, codificação e remoção de colunas. Se esse dataframe for reutilizado diretamente, torna-se mais difícil identificar se uma melhoria futura veio realmente de uma nova variável ou de uma transformação herdada.

Partir do conjunto de dados processado permite que cada experimento defina claramente seu próprio conjunto de atributos:

* quais colunas são removidas;
* como `tipo_de_acidente` é agrupado;
* se `trecho` é mantido ou removido;
* como `km_ajustado` e `km_bin` são representados;
* como os atributos temporais são criados;
* quais variáveis categóricas são codificadas;
* quais variáveis numéricas são escaladas.

O dataset processado é o ponto de partida correto porque já contém as principais decisões de limpeza da etapa de preparação dos dados, incluindo a correção da lógica dos quilômetros e a variável alvo `acidente_fatal`. As transformações específicas de modelagem devem permanecer dentro do script de modelagem ou de funções reutilizáveis.

### Resultados de Linha de Base

A primeira linha de base foi um `DummyClassifier` utilizando `most_frequent`.

Resultado:

```text
accuracy = 0.9843
```

A distribuição do conjunto de teste foi:

```text
0 = 98.43%
1 = 1.57%
```

Esse resultado mostrou que a acurácia é uma métrica enganosa para este problema. Um modelo pode atingir uma acurácia muito alta simplesmente prevendo quase todos os acidentes como não fatais. Portanto, a avaliação deve focar principalmente na classe fatal.

Métricas importantes:

* precision da classe `1`;
* recall da classe `1`;
* F1-score da classe `1`;
* matriz de confusão;
* falsos positivos;
* falsos negativos;
* verdadeiros positivos.

### Linha de Base com Regressão Logística

A primeira modelagem real utilizou Regressão Logística com:

```text
class_weight = balanced
```

Após a padronização das variáveis numéricas, o aviso de convergência desapareceu, mas as métricas mudaram apenas ligeiramente. Isso era esperado: a escala auxilia a otimização, mas não resolve necessariamente o desbalanceamento de classes.

Em um limiar de decisão (`threshold`) de `0.5`, o modelo produziu:

```text
accuracy = 0.8008
precision fatal = 0.06
recall fatal = 0.74
F1 fatal = 0.10
```

Matriz de confusão:

```text
[[21713  5371]
 [  111   321]]
```

Interpretação:

* o modelo identificou 321 dos 432 acidentes fatais;
* o recall foi elevado em comparação ao baseline;
* a precision foi muito baixa;
* o modelo gerou muitos falsos positivos.

Testes de threshold mostraram que limiares mais altos reduziram os falsos positivos e melhoraram o F1-score da classe fatal.

Após uma separação mais cuidadosa das categorias relacionadas a atropelamentos, o resultado da Regressão Logística com threshold `0.8` foi:

```text
accuracy = 0.9614
precision fatal = 0.18
recall fatal = 0.41
F1 fatal = 0.25
```

Matriz de confusão:

```text
[[26277   807]
 [  255   177]]
```

Esta passou a ser a melhor configuração da Regressão Logística até o momento.

### Linha de Base com Árvore de Decisão

Uma primeira `DecisionTreeClassifier` com:

```text
class_weight = balanced
```

produziu:

```text
accuracy = 0.9750
precision fatal = 0.17
recall fatal = 0.15
F1 fatal = 0.15
```

Matriz de confusão:

```text
[[26766   318]
 [  369    63]]
```

A árvore foi mais conservadora do que a Regressão Logística. Ela gerou menos falsos positivos, mas deixou de identificar a maior parte dos acidentes fatais.

Testes adicionais foram realizados com:

```text
max_depth = 4, 6, 8, 10
min_samples_leaf = 20, 50, 100
```

A melhor configuração testada alcançou aproximadamente:

```text
test F1 fatal = 0.110
```

Mesmo assim, o desempenho permaneceu inferior ao da Regressão Logística com ajuste de threshold.

### Linha de Base com Random Forest

A primeira `RandomForestClassifier` com:

```text
class_weight = balanced
```

e threshold padrão de `0.5` produziu:

```text
accuracy = 0.9839
precision fatal = 0.39
recall fatal = 0.04
F1 fatal = 0.07
```

Matriz de confusão:

```text
[[27057    27]
 [  415    17]]
```

Esse modelo foi extremamente conservador. Ele quase nunca previu a classe fatal.

Após testes de thresholds e hiperparâmetros, a melhor configuração de Random Forest em termos de F1-score da classe fatal foi:

```text
class_weight = balanced
max_depth = None
min_samples_leaf = 1
threshold = 0.20
```

Resultado:

```text
accuracy = 0.9811
precision fatal = 0.3481
recall fatal = 0.2361
F1 fatal = 0.2814
false positives = 191
false negatives = 330
true positives = 102
```

Esse passou a ser o melhor modelo até o momento considerando o F1-score da classe fatal. Em comparação com a Regressão Logística, apresenta maior precisão e menos falsos positivos, porém identifica uma parcela menor dos acidentes fatais.

### Interpretação Atual dos Modelos

Os modelos atuais apresentam diferentes compromissos entre precisão e sensibilidade:

* O DummyClassifier demonstra que acurácia, isoladamente, não é suficiente para avaliar o problema.
* A Regressão Logística identifica mais acidentes fatais, mas produz muitos falsos positivos.
* A Árvore de Decisão é interpretável, porém apresenta desempenho fraco para a classe fatal.
* O Random Forest possui atualmente o melhor F1-score para a classe fatal, mas apresenta recall inferior ao da Regressão Logística.

Esses resultados indicam que o próximo passo não deve focar apenas em ajustes de algoritmos. Melhorias na representação dos atributos podem gerar ganhos mais relevantes.

### Próxima Direção de Engenharia de Atributos

A próxima iteração de modelagem irá avaliar se uma engenharia de atributos mais cuidadosa melhora a capacidade de prever acidentes fatais.

Testes planejados:

#### 1. Remover `trecho`

Justificativa:

`km_ajustado` e `km_bin_id` já representam a posição na rodovia de forma mais precisa. A variável `trecho` pode ser redundante e contribuir com pouca informação adicional.

#### 2. Separar categorias de colisão

O agrupamento atual pode estar excessivamente amplo. Os tipos de colisão devem ser separados em categorias mais específicas, como:

```text
colisao_traseira
colisao_frontal
colisao_lateral
colisao_transversal
colisao_outros
```

Justificativa:

Colisões traseiras e frontais apresentam padrões de severidade diferentes. Agrupar todas as colisões em uma única categoria pode ocultar sinais importantes relacionados ao risco de fatalidade.

#### 3. Testar melhores atributos espaciais

Possíveis alternativas:

* manter `km_ajustado`;
* manter ou remover `km_bin_id`;
* utilizar codificação one-hot para os intervalos de quilometragem;
* criar indicadores de hotspot, como `zona_inicio`, `zona_final` ou `zona_hotspot`.

Justificativa:

A análise exploratória mostrou forte concentração espacial dos acidentes fatais, especialmente no início e no final da sequência corrigida da rodovia.

#### 4. Simplificar atributos temporais

Os atributos temporais atuais incluem:

```text
hora_sin
hora_cos
hour
bin_hora
```

Essas variáveis podem ser parcialmente redundantes.

Testes futuros devem comparar:

* apenas representação cíclica (`hora_sin` e `hora_cos`);
* apenas representação por faixas (`bin_hora`);
* categorias de período do dia.

#### 5. Adicionar atributos de interação entre veículos

Possíveis atributos:

* `total_veiculos`;
* `envolve_caminhao`;
* `envolve_moto`;
* interações entre tipo de acidente e envolvimento de determinados veículos.

Justificativa:

A análise exploratória mostrou que o envolvimento de veículos varia significativamente conforme o tipo de acidente, especialmente em ocorrências com caminhões, motocicletas e atropelamentos.

### Disciplina Experimental

Os testes futuros devem alterar apenas uma decisão de engenharia de atributos por vez.

Processo recomendado:

```text
1. Definir um conjunto de atributos de referência.
2. Treinar os modelos selecionados com thresholds fixos.
3. Salvar as métricas.
4. Alterar uma única decisão de engenharia de atributos.
5. Treinar novamente.
6. Comparar precision, recall e F1 da classe fatal.
```

A principal tabela de comparação deve utilizar:

```text
model
feature_version
threshold
precision_fatal
recall_fatal
f1_fatal
false_positives
false_negatives
true_positives
```

O próximo passo prático é criar uma versão mais limpa dos experimentos de modelagem, iniciando diretamente de `acidentes_tratados.csv`, reconstruindo o conjunto de atributos em cada execução e comparando os modelos de referência sob mudanças controladas de engenharia de atributos.
