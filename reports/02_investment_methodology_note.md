# Investment Analysis Methodology Note

This note documents the change in reasoning before starting the investment analysis step. The initial idea was to compare investment in year `t` with fatal crashes in year `t + 1`. This lagged comparison is still useful as an exploratory test, but it is probably too simple to explain the behavior of roadway safety after infrastructure investment.

Roadway investments do not behave like one-year interventions whose effect disappears in the following year. When investment is related to pavement rehabilitation, new lanes, shoulders, geometry improvements, safety devices, drainage, signage, or other physical improvements, part of the benefit may persist for several years. In this context, the effect may appear less as a direct annual response and more as a change in the baseline level of risk.

This matters because the fatal crash series suggests a possible change of level after the high-investment period around 2011-2014. In nominal terms, Autopista Fernao Dias had investments of R$ 225.8 million in 2011, R$ 242.7 million in 2012, R$ 271.3 million in 2013, and R$ 239.9 million in 2014. After correction to 2024 reais, these become approximately R$ 471.2 million, R$ 478.4 million, R$ 505.0 million, and R$ 419.6 million, respectively. These are among the highest corrected investment values in the available series.

The fatal crash counts also change substantially around this period. From 2010 to 2013, the annual fatal crash counts were 183, 190, 179, and 186, with an average of 184.5 fatal crashes per year. From 2014 to 2025, the average falls to 116.5 fatal crashes per year. A shorter intermediate window from 2014 to 2017 has an average of 128.0, while the period from 2018 to 2025 has an average of 110.75.

This pattern does not prove that the investment caused the reduction. Other factors may have changed at the same time, including traffic volume, enforcement, vehicle fleet composition, operational procedures, economic activity, reporting practices, concession requirements, or specific road safety interventions not separately identified in the investment dataset. However, the pattern suggests that a simple one-year lag may miss the more relevant structure of the problem.

The year 2017 is an important example of this limitation. Corrected investment in 2017 was approximately R$ 286.4 million, lower than the corrected values observed in the earlier high-investment period. Even so, fatal crashes did not return to the 2010-2013 level. This is consistent with the idea that some infrastructure improvements may create persistent safety effects. Once a road segment receives a structural improvement, the benefit may remain, while later spending may be more related to maintenance, conservation, signage, and preserving service quality.

Because of this, the investment analysis should not rely only on the question:

```
Does investment in year t reduce fatal crashes in year t + 1?
```

A better exploratory framing is:

```
Are high-investment periods associated with later changes in the fatal crash baseline?
```

or:

```
Does the series show a sustained change in fatal crashes after major investment cycles?
```

The next analysis step should therefore include multiple views:

1. Annual corrected investment.
2. Annual fatal crashes.
3. A combined chart to inspect possible changes of level.
4. A simple one-year lag test, treated only as an exploratory baseline.
5. Rolling or accumulated investment over three years.
6. Before-and-after comparisons around the highest-investment years.

The main methodological caution is that investment data is annual and aggregated by concessionaire. It does not identify the exact location, type, or purpose of each investment. Therefore, any relationship between investment and fatal crash reduction must be interpreted as exploratory association, not causal proof.


# Nota Metodológica da Análise de Investimentos - PT-BR
Esta nota documenta a mudança de raciocínio antes do início da etapa de análise de investimentos. A ideia inicial era comparar o investimento no ano t com acidentes fatais no ano t + 1. Essa comparação defasada ainda é útil como teste exploratório, mas provavelmente é simples demais para explicar o comportamento da segurança viária após investimentos em infraestrutura.

Investimentos rodoviários não se comportam como intervenções de um único ano cujo efeito desaparece no ano seguinte. Quando o investimento está relacionado à recuperação de pavimento, novas faixas, acostamentos, melhorias geométricas, dispositivos de segurança, drenagem, sinalização ou outras melhorias físicas, parte do benefício pode persistir por vários anos. Nesse contexto, o efeito pode aparecer menos como uma resposta anual direta e mais como uma mudança no nível de base do risco.

Isso é importante porque a série de acidentes fatais sugere uma possível mudança de nível após o período de altos investimentos entre 2011 e 2014. Em termos nominais, a Autopista Fernão Dias teve investimentos de R$ 225,8 milhões em 2011, R$ 242,7 milhões em 2012, R$ 271,3 milhões em 2013 e R$ 239,9 milhões em 2014. Após correção para valores de 2024, esses montantes se tornam aproximadamente R$ 471,2 milhões, R$ 478,4 milhões, R$ 505,0 milhões e R$ 419,6 milhões, respectivamente. Esses estão entre os maiores valores corrigidos de investimento em toda a série disponível.

As contagens de acidentes fatais também mudam substancialmente ao redor desse período. De 2010 a 2013, os totais anuais de acidentes fatais foram 183, 190, 179 e 186, com média de 184,5 acidentes fatais por ano. De 2014 a 2025, a média cai para 116,5 acidentes fatais por ano. Uma janela intermediária mais curta, de 2014 a 2017, apresenta média de 128,0, enquanto o período de 2018 a 2025 apresenta média de 110,75.

Esse padrão não prova que o investimento causou a redução. Outros fatores podem ter mudado ao mesmo tempo, incluindo volume de tráfego, fiscalização, composição da frota, procedimentos operacionais, atividade econômica, práticas de registro, exigências da concessão ou intervenções específicas de segurança viária não identificadas separadamente na base de investimentos. Ainda assim, o padrão sugere que uma simples defasagem de um ano pode deixar de captar a estrutura mais relevante do problema.

O ano de 2017 é um exemplo importante dessa limitação. O investimento corrigido em 2017 foi de aproximadamente R$ 286,4 milhões, menor que os valores corrigidos observados no período anterior de altos investimentos. Mesmo assim, os acidentes fatais não retornaram ao nível observado entre 2010 e 2013. Isso é consistente com a ideia de que algumas melhorias de infraestrutura podem gerar efeitos persistentes na segurança. Uma vez que um trecho recebe uma melhoria estrutural, o benefício pode permanecer, enquanto gastos posteriores podem estar mais relacionados à manutenção, conservação, sinalização e preservação da qualidade operacional.

Por causa disso, a análise de investimentos não deve depender apenas da pergunta:

```
O investimento no ano t reduz acidentes fatais no ano t + 1?
```

Uma formulação exploratória mais adequada é:

```
Períodos de alto investimento estão associados a mudanças posteriores no nível de base dos acidentes fatais?
```

ou:

```
A série apresenta uma mudança sustentada nos acidentes fatais após grandes ciclos de investimento?
```

A próxima etapa da análise deve, portanto, incluir múltiplas perspectivas:

Investimento anual corrigido.
Acidentes fatais anuais.
Um gráfico combinado para inspecionar possíveis mudanças de nível.
Um teste simples de defasagem de um ano, tratado apenas como linha de base exploratória.
Investimento acumulado ou móvel em três anos.
Comparações antes e depois dos anos de maior investimento.

A principal cautela metodológica é que os dados de investimento são anuais e agregados por concessionária. Eles não identificam a localização exata, o tipo ou a finalidade de cada investimento. Portanto, qualquer relação entre investimento e redução de acidentes fatais deve ser interpretada como associação exploratória, e não como prova causal.