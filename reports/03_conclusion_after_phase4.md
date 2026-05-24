# Conclusion After Phase 4

The investment analysis helped refine the central question of the project. The original economic question was:

```
How much investment would be necessary to reduce the number of fatal crashes?
```

After correcting investment values for inflation and comparing them with fatal crash trends, the analysis suggests that this question should not be answered with a single fixed value. The relationship between investment and fatal crashes appears to depend on the phase of the concession and on the type of intervention made in the roadway.

In the initial period, especially around 2011-2014, corrected investments were high and coincided with a later reduction in the fatal-crash baseline. Fatal crashes were around 180-190 per year in the early period and later moved to a lower range. This pattern is consistent with the hypothesis that structural works can create persistent effects. Investments in pavement rehabilitation, new lanes, shoulders, access improvements, safety devices, drainage, signage or geometric changes may alter the physical and operational condition of the road for several years.

For that reason, the initial investment cycle should not be interpreted only as a one-year lag effect. It is more appropriate to treat it as a possible change in the structural risk level of the highway. A large infrastructure investment may reduce risk beyond the year immediately after the investment.

In the mature and final phases of the concession, the annual relationship appears more visible. After the main structural change, investment may be more related to maintenance, conservation, localized improvements, signage, road operation and preservation of the achieved safety level. In this phase, higher investment in one year appears more plausibly associated with fewer fatal crashes in the following year, while lower investment may be associated with deterioration in the following year.

However, the data is not sufficient to estimate a reliable value such as:

```
R$ X million reduce Y fatal crashes.
```

The investment dataset has few annual observations and is aggregated by concessionaire. It does not identify exactly where the money was applied, which type of work was executed, how much was maintenance versus structural work, or which road segments were affected. Other factors may also influence fatal crashes, such as traffic volume, enforcement, economic activity, weather, vehicle fleet composition, reporting practices and local road geometry.

Therefore, the conclusion of Phase 4 is exploratory:

```
The data suggests an association between investment and fatal-crash reduction, especially through two mechanisms: structural investment capable of changing the risk baseline, and recurring investment that may help maintain or improve that baseline. The available data does not allow a causal estimate of how much investment is necessary to reduce a specific number of fatal crashes.
```

This is not a failure of the analysis. It is a methodological limitation of the available data. The annual investment series is too small to support a robust predictive model for investment scenarios.

Because of that, the next modeling step will move from annual investment data to occurrence-level crash data. The crash dataset has many more observations and allows a better machine learning question:

```
Given the characteristics of a crash occurrence, can a model predict whether it will be fatal?
```

The new target variable will be:

```
acidente_fatal
```

Potential predictors include:

- crash type;
- vehicles involved;
- corrected kilometer range;
- road direction;
- road segment;
- year and month;
- time of day or time period, if properly engineered.

The modeling step must avoid data leakage. Variables such as `mortos` cannot be used as predictors because they define the target. Injury outcome variables must also be handled carefully, because they may describe consequences of the crash rather than information available before or at the time of classification.

The next stage will therefore focus on classification models, starting with simple and interpretable baselines before testing more complex models. Because fatal crashes are rare compared with non-fatal crashes, accuracy alone will not be enough. The analysis should prioritize metrics such as recall, precision, F1-score, confusion matrix and possibly PR-AUC or ROC-AUC for the fatal-crash class.

The final direction after Phase 4 is:

```
Use the investment analysis as an exploratory contextual finding, and use occurrence-level crash data for predictive modeling of fatal-crash risk.
```


# Conclusão Após a Fase 4 - PT-BR
A análise de investimento ajudou a refinar a questão central do projeto. A pergunta econômica original era:

```
Quanto investimento seria necessário para reduzir o número de acidentes fatais?
```

Após corrigir os valores de investimento pela inflação e compará-los com as tendências de acidentes fatais, a análise sugere que essa pergunta não deve ser respondida com um único valor fixo. A relação entre investimento e acidentes fatais parece depender da fase da concessão e do tipo de intervenção realizada na rodovia.

No período inicial, especialmente entre 2011 e 2014, os investimentos corrigidos foram elevados e coincidiram com uma redução posterior do patamar de acidentes fatais. Os acidentes fatais estavam em torno de 180–190 por ano no período inicial e depois passaram para uma faixa mais baixa. Esse padrão é consistente com a hipótese de que obras estruturais podem gerar efeitos persistentes. Investimentos em recuperação de pavimento, novas faixas, acostamentos, melhorias de acesso, dispositivos de segurança, drenagem, sinalização ou mudanças geométricas podem alterar as condições físicas e operacionais da rodovia por vários anos.

Por esse motivo, o ciclo inicial de investimentos não deve ser interpretado apenas como um efeito de defasagem de um ano. É mais apropriado tratá-lo como uma possível mudança no nível estrutural de risco da rodovia. Um grande investimento em infraestrutura pode reduzir o risco além do ano imediatamente posterior ao investimento.

Nas fases madura e final da concessão, a relação anual parece mais visível. Após a principal mudança estrutural, o investimento pode estar mais relacionado à manutenção, conservação, melhorias localizadas, sinalização, operação da rodovia e preservação do nível de segurança alcançado. Nessa fase, um investimento maior em um ano parece estar mais plausivelmente associado a menos acidentes fatais no ano seguinte, enquanto um investimento menor pode estar associado a deterioração no ano posterior.

No entanto, os dados não são suficientes para estimar um valor confiável como:

```
R$ X milhões reduzem Y acidentes fatais.
```

O conjunto de dados de investimento possui poucas observações anuais e está agregado por concessionária. Ele não identifica exatamente onde o dinheiro foi aplicado, qual tipo de obra foi executado, quanto correspondeu à manutenção versus obras estruturais, nem quais segmentos da rodovia foram afetados. Outros fatores também podem influenciar os acidentes fatais, como volume de tráfego, fiscalização, atividade econômica, condições climáticas, composição da frota de veículos, práticas de registro e geometria local da via.

Portanto, a conclusão da Fase 4 é exploratória:

```
Os dados sugerem uma associação entre investimento e redução de acidentes fatais, especialmente por meio de dois mecanismos: investimentos estruturais capazes de alterar o nível de risco da rodovia e investimentos recorrentes que podem ajudar a manter ou melhorar esse nível. Os dados disponíveis não permitem uma estimativa causal de quanto investimento é necessário para reduzir um número específico de acidentes fatais.
```

Isso não representa uma falha da análise. Trata-se de uma limitação metodológica dos dados disponíveis. A série anual de investimentos é pequena demais para sustentar um modelo preditivo robusto para cenários de investimento.

Por causa disso, a próxima etapa da modelagem passará dos dados anuais de investimento para dados de acidentes em nível de ocorrência. O conjunto de dados de acidentes possui muito mais observações e permite uma pergunta de aprendizado de máquina mais adequada:

```
Dadas as características de uma ocorrência de acidente, um modelo consegue prever se o acidente será fatal?
```

A nova variável alvo será:

```
acidente_fatal
```

Os possíveis preditores incluem:

tipo de acidente;
veículos envolvidos;
faixa quilométrica corrigida;
sentido da rodovia;
segmento da rodovia;
ano e mês;
horário ou período do dia, se adequadamente tratado por engenharia de atributos.

A etapa de modelagem deve evitar vazamento de dados (data leakage). Variáveis como mortos não podem ser usadas como preditoras, porque definem a variável alvo. Variáveis relacionadas ao desfecho de lesões também devem ser tratadas com cuidado, pois podem descrever consequências do acidente, e não informações disponíveis antes ou no momento da classificação.

A próxima etapa, portanto, será focada em modelos de classificação, começando por modelos simples e interpretáveis antes de testar modelos mais complexos. Como acidentes fatais são raros em comparação com acidentes não fatais, a acurácia sozinha não será suficiente. A análise deve priorizar métricas como recall, precision, F1-score, matriz de confusão e possivelmente PR-AUC ou ROC-AUC para a classe de acidentes fatais.

A direção final após a Fase 4 é:

```
Usar a análise de investimentos como um resultado exploratório contextual e utilizar dados de acidentes em nível de ocorrência para a modelagem preditiva do risco de acidentes fatais.
```