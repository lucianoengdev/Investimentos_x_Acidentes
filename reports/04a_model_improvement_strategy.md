# Model Improvement Strategy - Development Note

This report documents the modeling improvement phase after the first machine learning baseline. It should be read as a development note: it records the reasoning process, the tested directions, the results observed so far, and the decision to focus the next modeling effort on a deeper Random Forest iteration.

The goal of this phase is not to prove that the model is final. The goal is to understand which modeling decisions are actually improving fatal-crash prediction and which ones are only adding complexity without meaningful gain.

## Modeling Question

The modeling question remains:

```text
Given the characteristics of a crash occurrence, can a model estimate whether the crash is likely to be fatal?
```

The target variable is:

```text
acidente_fatal
```

This is a highly imbalanced binary classification problem:

```text
0 = non-fatal crash
1 = fatal crash
```

Because fatal crashes are rare, the main evaluation focus is not overall accuracy. The most important metrics are:

* precision for the fatal class;
* recall for the fatal class;
* F1-score for the fatal class;
* false positives;
* false negatives;
* true positives.

## Improvement Process Followed

After the initial baseline, the modeling work moved from simply comparing algorithms to testing feature representation and decision thresholds in a controlled way.

The main development rule was:

```text
Change one modeling decision at a time, compare the fatal-class metrics, and keep only changes that improve the model in a meaningful way.
```

The reference models used during the experiments were:

* Logistic Regression with `class_weight='balanced'`;
* Decision Tree with `class_weight='balanced'`;
* Random Forest with class balancing and manual threshold selection.

The Random Forest gradually became the strongest reference model because it produced the best fatal-class F1-score while keeping a better precision-recall balance than the other models.

## Feature Engineering Tests

Several feature-engineering directions were tested.

### Removing `trecho`

The feature `trecho` was removed because the corrected kilometer variable already represents the spatial position of the crash more directly.

This change did not hurt the model and supported the interpretation that `trecho` was mostly redundant once `km_ajustado` was available.

### Separating Collision Types

The broad `colisao` category was separated into more specific categories such as:

```text
colisao_frontal
colisao_lateral
colisao_transversal
colisao_traseira
```

This improved the representation of crash type. It helped the models distinguish between collision mechanisms that likely have different fatality patterns.

This change was kept.

### Time Features

Different time representations were tested:

* `hora_sin` and `hora_cos`;
* hour-based columns;
* binned time periods such as morning, afternoon, night and dawn.

The cyclical representation using `hora_sin` and `hora_cos` was kept because it represents the daily cycle without creating artificial distance between nearby times such as 23:00 and 00:00.

The simpler time-period representation was easier to interpret, but it did not improve the model.

### Spatial Features

The model tested:

* `km_ajustado`;
* `km_bin_id`;
* kilometer-bin one-hot encoding;
* hotspot flags based on the exploratory analysis.

The best result came from keeping `km_ajustado` as the main spatial feature. `km_bin_id` and one-hot kilometer bins did not improve the model enough to justify the added complexity.

This result is consistent with the earlier spatial correction: once the kilometer sequence was corrected, the continuous adjusted kilometer variable became one of the most useful predictors.

### Year and Month Encoding

Year and month were tested as one-hot encoded variables.

This did not improve the models. The working version therefore keeps year and month as numeric features rather than expanding them into several dummy columns.

### Vehicle Grouping and Interaction Features

Several vehicle-related ideas were tested:

* grouping heavy vehicles;
* grouping vulnerable users such as motorcycles and bicycles;
* counting vehicle types involved;
* creating flags such as `tem_pesado` and `tem_vulneravel`;
* testing interactions between vehicle type and crash type.

The interaction tests included combinations such as:

```text
caminhao + atropelamento
moto + colisao frontal
moto + colisao traseira
automovel + colisao traseira
```

These interactions did not improve performance consistently. Some small gains appeared in one model, but they were not strong enough across the full model comparison. In some cases, they improved recall while hurting precision or F1-score.

The conclusion was to avoid adding these interaction features for now. They increase model complexity without enough evidence of stable improvement.

## Current Best Random Forest Results

The strongest model before deeper tuning was a Random Forest using:

```text
n_estimators = 300
min_samples_leaf = 2
max_features = sqrt
class_weight = balanced
threshold = 0.35
```

Result:

```text
accuracy = 0.980048
precision fatal = 0.349614
recall fatal = 0.314815
F1 fatal = 0.331303
false positives = 253
false negatives = 296
true positives = 136
```

This became the best general model by fatal-class F1-score at that stage.

Compared with the earlier Random Forest, this version captured more fatal crashes and reduced false negatives, with only a moderate increase in false positives.

## Undersampling Test

Because fatal crashes are rare, a simple undersampling strategy was tested.

The best tested setup kept all fatal crashes in the training set and reduced the non-fatal class to:

```text
15 non-fatal crashes for each fatal crash
```

This model used the same Random Forest structure:

```text
n_estimators = 300
min_samples_leaf = 2
max_features = sqrt
threshold = 0.35
```

Result:

```text
accuracy = 0.978195
precision fatal = 0.322785
recall fatal = 0.354167
F1 fatal = 0.337748
false positives = 321
false negatives = 279
true positives = 153
```

This was the best fatal-class F1-score observed so far.

The trade-off is clear:

* the undersampled model finds more fatal crashes;
* it reduces false negatives;
* it increases false positives;
* it lowers precision slightly;
* it improves recall and F1-score.

This makes the undersampled Random Forest a better candidate when the priority is to identify more potentially fatal occurrences, accepting more false alarms as a cost.

## Why Focus on Random Forest Now?

The decision to focus on Random Forest is methodological, not just practical.

Logistic Regression remains useful as a baseline because it is simple and interpretable. However, it consistently produced many false positives when tuned for fatal-crash recall. It can capture more fatal crashes, but the precision-recall balance remains weak.

Decision Tree is interpretable, but it underperformed for the fatal class. It usually missed too many fatal crashes and did not compete with the stronger models.

Random Forest has produced the best balance so far:

* stronger fatal-class F1-score;
* better precision than Logistic Regression;
* better recall than a conservative default forest;
* ability to handle nonlinear relationships;
* ability to use spatial, temporal, vehicle and crash-type features together;
* more stable behavior than a single Decision Tree.

Additional algorithms such as Extra Trees, Gradient Boosting and HistGradientBoosting were also tested in exploratory form, but they did not outperform the tuned Random Forest result in the current setup.

For that reason, the best next step is not to keep testing many algorithms superficially. The better path is to focus on one promising model family and improve it more carefully.

## Current Modeling Choice

The current best development direction is:

```text
Use Random Forest as the main model for deeper tuning.
```

Two Random Forest versions should be kept for comparison:

### Conservative Random Forest

```text
class_weight = balanced
n_estimators = 300
min_samples_leaf = 2
threshold = 0.35
```

This version has higher precision and fewer false positives.

### Recall-Oriented Random Forest

```text
undersampling = 15 non-fatal crashes per fatal crash
n_estimators = 300
min_samples_leaf = 2
threshold = 0.35
```

This version has higher recall and better fatal-class F1-score, but produces more false positives.

The choice between these two models depends on the operational goal:

```text
If false alarms are expensive, use the conservative Random Forest.
If missing fatal-risk crashes is more serious, use the undersampled Random Forest.
```

## Development Observation

At this stage, the main learning is that feature engineering alone has reached a point of diminishing returns. Many reasonable features were tested, but only a few improved the model:

* separating collision types;
* keeping `km_ajustado`;
* using cyclical time features;
* tuning Random Forest parameters;
* testing class resampling.

The strongest improvement came not from adding more explanatory columns, but from changing how the model handles class imbalance and from tuning the Random Forest decision behavior.

This suggests that the next phase should be less about creating more manual features and more about:

* validating the chosen model more rigorously;
* testing cross-validation instead of relying on one train-test split;
* analyzing precision-recall curves;
* testing threshold selection systematically;
* checking feature importance and permutation importance;
* testing whether the model is stable across years or concession phases;
* avoiding overfitting to the current test split.

## Recommended Next Step

The best next path is to continue with Random Forest, but move from exploratory testing to validation.

Recommended next actions:

```text
1. Keep the current feature set as the modeling reference.
2. Keep both Random Forest versions: conservative and recall-oriented.
3. Add cross-validation focused on fatal-class F1-score and recall.
4. Compare thresholds using a precision-recall curve.
5. Inspect feature importance and permutation importance.
6. Test temporal validation, such as training on earlier years and testing on later years.
```

This is a better path than continuing to add many small manual interaction features. The current evidence suggests that the model can still improve, but the next improvement should come from stronger validation and more controlled tuning, not from uncontrolled feature expansion.

---

# Estrategia de Melhoria do Modelo - Nota de Desenvolvimento - PT-BR

Este report documenta a fase de melhoria da modelagem apos a primeira linha de base de machine learning. Ele deve ser lido como uma nota de desenvolvimento: registra a linha de raciocinio, as direcoes testadas, os resultados observados ate aqui e a decisao de concentrar o proximo esforco em uma iteracao mais aprofundada com Random Forest.

O objetivo desta fase nao e provar que o modelo esta finalizado. O objetivo e entender quais decisoes de modelagem estao realmente melhorando a previsao de acidentes fatais e quais apenas adicionam complexidade sem ganho relevante.

## Pergunta da Modelagem

A pergunta de modelagem continua sendo:

```text
Dadas as caracteristicas de uma ocorrencia de acidente, um modelo consegue estimar se o acidente tem maior probabilidade de ser fatal?
```

A variavel alvo e:

```text
acidente_fatal
```

Este e um problema de classificacao binaria altamente desbalanceado:

```text
0 = acidente nao fatal
1 = acidente fatal
```

Como acidentes fatais sao raros, o foco principal da avaliacao nao e a acuracia geral. As metricas mais importantes sao:

* precision da classe fatal;
* recall da classe fatal;
* F1-score da classe fatal;
* falsos positivos;
* falsos negativos;
* verdadeiros positivos.

## Processo de Melhoria Seguido

Depois da primeira linha de base, a modelagem deixou de ser apenas uma comparacao simples entre algoritmos e passou a testar representacoes de variaveis e thresholds de decisao de forma mais controlada.

A regra principal de desenvolvimento foi:

```text
Alterar uma decisao de modelagem por vez, comparar as metricas da classe fatal e manter apenas as mudancas que melhoram o modelo de forma relevante.
```

Os modelos de referencia usados durante os experimentos foram:

* Regressao Logistica com `class_weight='balanced'`;
* Decision Tree com `class_weight='balanced'`;
* Random Forest com balanceamento de classe e escolha manual de threshold.

A Random Forest passou gradualmente a ser o modelo de referencia mais forte porque apresentou o melhor F1-score da classe fatal, mantendo um equilibrio melhor entre precision e recall do que os demais modelos.

## Testes de Engenharia de Atributos

Varias direcoes de engenharia de atributos foram testadas.

### Remocao de `trecho`

A variavel `trecho` foi removida porque a variavel de quilometro corrigido ja representa a posicao espacial do acidente de forma mais direta.

Essa alteracao nao prejudicou o modelo e reforcou a interpretacao de que `trecho` era uma informacao em grande parte redundante quando `km_ajustado` estava disponivel.

### Separacao dos Tipos de Colisao

A categoria ampla `colisao` foi separada em categorias mais especificas, como:

```text
colisao_frontal
colisao_lateral
colisao_transversal
colisao_traseira
```

Essa mudanca melhorou a representacao do tipo de acidente. Ela ajudou os modelos a diferenciar mecanismos de colisao que provavelmente possuem padroes diferentes de fatalidade.

Essa alteracao foi mantida.

### Variaveis de Horario

Foram testadas diferentes representacoes de horario:

* `hora_sin` e `hora_cos`;
* colunas baseadas na hora;
* periodos do dia, como manha, tarde, noite e madrugada.

A representacao ciclica com `hora_sin` e `hora_cos` foi mantida porque representa o ciclo diario sem criar uma distancia artificial entre horarios proximos, como 23:00 e 00:00.

A representacao por periodo do dia era mais facil de interpretar, mas nao melhorou o modelo.

### Variaveis Espaciais

O modelo testou:

* `km_ajustado`;
* `km_bin_id`;
* one-hot encoding das faixas de quilometro;
* flags de hotspot baseadas na analise exploratoria.

O melhor resultado veio de manter `km_ajustado` como principal variavel espacial. `km_bin_id` e one-hot encoding das faixas de quilometro nao melhoraram o modelo o suficiente para justificar a complexidade adicional.

Esse resultado e coerente com a correcao espacial feita anteriormente: depois que a sequencia de quilometros foi corrigida, a variavel continua `km_ajustado` se tornou um dos preditores mais uteis.

### Codificacao de Ano e Mes

Ano e mes foram testados com one-hot encoding.

Essa alteracao nao melhorou os modelos. A versao atual, portanto, mantem ano e mes como variaveis numericas, em vez de expandi-las em varias colunas dummy.

### Agrupamento de Veiculos e Interacoes

Foram testadas varias ideias relacionadas aos veiculos:

* agrupamento de veiculos pesados;
* agrupamento de usuarios vulneraveis, como motos e bicicletas;
* contagem de tipos de veiculos envolvidos;
* flags como `tem_pesado` e `tem_vulneravel`;
* interacoes entre tipo de veiculo e tipo de acidente.

Os testes de interacao incluiram combinacoes como:

```text
caminhao + atropelamento
moto + colisao frontal
moto + colisao traseira
automovel + colisao traseira
```

Essas interacoes nao melhoraram o desempenho de forma consistente. Alguns pequenos ganhos apareceram em um modelo especifico, mas nao foram fortes o suficiente na comparacao geral. Em alguns casos, aumentavam recall enquanto prejudicavam precision ou F1-score.

A conclusao foi nao adicionar essas interacoes por enquanto. Elas aumentam a complexidade do modelo sem evidencia suficiente de ganho estavel.

## Melhor Resultado Atual com Random Forest

O modelo mais forte antes do teste de reamostragem foi uma Random Forest com:

```text
n_estimators = 300
min_samples_leaf = 2
max_features = sqrt
class_weight = balanced
threshold = 0.35
```

Resultado:

```text
accuracy = 0.980048
precision fatal = 0.349614
recall fatal = 0.314815
F1 fatal = 0.331303
falsos positivos = 253
falsos negativos = 296
verdadeiros positivos = 136
```

Esse passou a ser o melhor modelo geral pelo F1-score da classe fatal naquele momento.

Comparado com a Random Forest anterior, essa versao capturou mais acidentes fatais e reduziu falsos negativos, com aumento moderado dos falsos positivos.

## Teste com Undersampling

Como acidentes fatais sao raros, foi testada uma estrategia simples de undersampling.

A melhor configuracao testada manteve todos os acidentes fatais no conjunto de treino e reduziu a classe nao fatal para:

```text
15 acidentes nao fatais para cada acidente fatal
```

Esse modelo usou a mesma estrutura de Random Forest:

```text
n_estimators = 300
min_samples_leaf = 2
max_features = sqrt
threshold = 0.35
```

Resultado:

```text
accuracy = 0.978195
precision fatal = 0.322785
recall fatal = 0.354167
F1 fatal = 0.337748
falsos positivos = 321
falsos negativos = 279
verdadeiros positivos = 153
```

Esse foi o melhor F1-score da classe fatal observado ate agora.

A troca e clara:

* o modelo com undersampling encontra mais acidentes fatais;
* reduz falsos negativos;
* aumenta falsos positivos;
* reduz levemente a precision;
* melhora recall e F1-score.

Isso torna a Random Forest com undersampling uma candidata melhor quando a prioridade e identificar mais ocorrencias potencialmente fatais, aceitando mais falsos alarmes como custo.

## Por Que Focar Agora em Random Forest?

A decisao de focar em Random Forest e metodologica, nao apenas pratica.

A Regressao Logistica continua util como baseline por ser simples e interpretavel. No entanto, ela produziu muitos falsos positivos quando ajustada para capturar mais acidentes fatais. Ela consegue aumentar recall, mas o equilibrio entre precision e recall continua fraco.

A Decision Tree e interpretavel, mas teve desempenho baixo para a classe fatal. Em geral, perdeu muitos acidentes fatais e nao competiu com os modelos mais fortes.

A Random Forest apresentou o melhor equilibrio ate agora:

* melhor F1-score da classe fatal;
* precision melhor que a Regressao Logistica;
* recall melhor que uma floresta conservadora padrao;
* capacidade de capturar relacoes nao lineares;
* capacidade de combinar variaveis espaciais, temporais, de veiculos e de tipo de acidente;
* comportamento mais estavel que uma unica arvore de decisao.

Algoritmos adicionais, como Extra Trees, Gradient Boosting e HistGradientBoosting, tambem foram testados de forma exploratoria, mas nao superaram o resultado da Random Forest ajustada na configuracao atual.

Por isso, o melhor proximo passo nao e continuar testando muitos algoritmos superficialmente. O caminho mais forte e escolher uma familia promissora de modelo e melhora-la de forma mais cuidadosa.

## Escolha Atual de Modelagem

A melhor direcao atual e:

```text
Usar Random Forest como modelo principal para tuning aprofundado.
```

Duas versoes de Random Forest devem ser mantidas para comparacao:

### Random Forest Conservadora

```text
class_weight = balanced
n_estimators = 300
min_samples_leaf = 2
threshold = 0.35
```

Essa versao possui maior precision e menos falsos positivos.

### Random Forest Orientada a Recall

```text
undersampling = 15 acidentes nao fatais para cada acidente fatal
n_estimators = 300
min_samples_leaf = 2
threshold = 0.35
```

Essa versao possui maior recall e melhor F1-score da classe fatal, mas gera mais falsos positivos.

A escolha entre essas duas versoes depende do objetivo operacional:

```text
Se falsos alarmes forem caros, usar a Random Forest conservadora.
Se perder acidentes com risco fatal for mais grave, usar a Random Forest com undersampling.
```

## Observacao de Desenvolvimento

Neste ponto, o principal aprendizado e que a engenharia de atributos comecou a apresentar retornos decrescentes. Muitas variaveis razoaveis foram testadas, mas poucas melhoraram o modelo:

* separar tipos de colisao;
* manter `km_ajustado`;
* usar horario ciclico;
* ajustar os parametros da Random Forest;
* testar reamostragem da classe.

A maior melhoria nao veio de adicionar mais colunas explicativas, mas de alterar a forma como o modelo lida com o desbalanceamento de classes e de ajustar o comportamento de decisao da Random Forest.

Isso sugere que a proxima fase deve ser menos focada em criar features manuais e mais focada em:

* validar melhor o modelo escolhido;
* testar validacao cruzada em vez de depender de uma unica divisao treino-teste;
* analisar curvas precision-recall;
* escolher threshold de forma sistematica;
* verificar feature importance e permutation importance;
* testar se o modelo se mantem estavel entre anos ou fases da concessao;
* evitar overfitting ao conjunto de teste atual.

## Proximo Passo Recomendado

O melhor caminho agora e continuar com Random Forest, mas sair da fase exploratoria e entrar em uma fase de validacao.

Proximas acoes recomendadas:

```text
1. Manter o conjunto atual de features como referencia da modelagem.
2. Manter as duas Random Forests: conservadora e orientada a recall.
3. Adicionar validacao cruzada focada em F1-score e recall da classe fatal.
4. Comparar thresholds usando curva precision-recall.
5. Inspecionar feature importance e permutation importance.
6. Testar validacao temporal, como treinar em anos anteriores e testar em anos posteriores.
```

Esse caminho e melhor do que continuar adicionando muitas pequenas interacoes manuais. A evidencia atual sugere que o modelo ainda pode melhorar, mas a proxima melhoria deve vir de validacao mais forte e tuning mais controlado, nao de expansao descontrolada de features.
