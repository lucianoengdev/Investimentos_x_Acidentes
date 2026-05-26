# BR-381 Fatal Crash Analysis

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge)
![scikit--learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)

**Current version:** `v0.4.0-alpha`

## What This Does

This project analyzes fatal crashes on the BR-381 / Autopista Fernao Dias highway and investigates how historical roadway investments may relate to fatal crash reduction.

Working features:

- Loads and cleans traffic crash data.
- Preserves the original kilometer marker and creates a corrected linear kilometer reference.
- Generates initial exploratory charts for fatal crashes, deaths, crash types and kilometer distribution.
- Adjusts historical investment values using annual IPCA inflation.
- Compares investment and fatal-crash patterns across concession phases.
- Documents hotspot, vehicle, crash-type and investment findings for a technical article.
- Keeps fatal-crash machine learning as a future complementary extension.

## What Is This?

This is a **Python data analysis and data science project**, not a web, desktop or mobile application.

The code is expected to produce:

- cleaned datasets in `data/processed/`;
- exploratory charts in `reports/figures/`;
- documented analysis scripts in `analysis/`;
- reusable data-processing and visualization functions in `src/`;
- a technical article about fatal crashes, investment and roadway safety;
- a future optional machine learning extension at occurrence level.

The main analytical metric is:

```text
fatal_crash = deaths > 0
```

In the original Portuguese dataset, this is equivalent to:

```text
acidente_fatal = mortos > 0
```

## Technologies Used

- **Python 3**: main programming language.
- **Pandas**: CSV loading, cleaning, grouping and aggregation.
- **NumPy**: numerical operations.
- **Matplotlib**: static chart generation.
- **Seaborn**: exploratory statistical visualizations.
- **scikit-learn**: planned future complementary machine learning extension.
- **Git / GitHub Desktop**: version control workflow.
- **VS Code**: development environment.

## Project Ambition

The goal is to build a complete transportation data project. The project is not intended to simply produce charts. It is intended to document a full reasoning process:

1. understand the raw data;
2. clean and validate the data;
3. generate exploratory charts;
4. interpret the results before modeling;
5. compare crash patterns with roadway investment;
6. write a public technical article with conclusions and limitations;
7. optionally extend the repository with occurrence-level fatal-crash classification.

The primary goal is to produce a transparent exploratory analysis of how fatal crashes, dangerous locations, crash types and historical investment patterns relate on the BR-381 / Fernao Dias corridor, while being careful not to claim causality without enough evidence. Machine learning is a later complementary path rather than part of the article's core scope.

## Project Stage

This project is a **work in progress**.

### Done

- [x] Step 1: Initial data preparation.
- [x] Step 1 update: Kilometer correction logic added.
- [x] Original `km` column preserved.
- [x] Corrected kilometer column created for spatial analysis.
- [x] Step 2: Initial exploratory charts generated.
- [x] Initial charts reviewed and used to detect a kilometer discontinuity.
- [x] Re-run Step 2 charts using the corrected kilometer reference.
- [x] Compare original kilometer distribution vs corrected kilometer distribution.
- [x] Step 3: Analyze hotspots by corrected kilometer, road direction and vehicle type.
- [x] Document the investment-analysis methodology before Step 4.
- [x] Step 4: Prepare and adjust investment data using IPCA inflation.
- [x] Compare corrected investment with fatal crashes using overall and phase-based exploratory views.
- [x] Document the conclusion and the limits of estimating investment effects.

### In Progress

- [ ] Write the technical article from the exploratory analysis and documented findings.

### Future Extension

- [ ] Define an occurrence-level fatal-crash classification question.
- [ ] Train and evaluate machine learning models as a complementary repository extension.

## Kilometer Correction

During the first exploratory charts, a visual gap appeared between approximately km 90 and km 477. This suggested that the kilometer field in the dataset does not represent a simple continuous sequence for the highway analysis.

The current working interpretation is:

- the physical/logical sequence starts near the high-kilometer section;
- after that section reaches the end, the dataset continues near km 0;
- therefore, spatial analysis needs a corrected linear kilometer reference.

Methodological rule:

- keep the original `km` column untouched;
- create a new corrected kilometer column;
- use the correct adjusted kilometer column;
- document the correction before drawing conclusions.

This matters because hotspot analysis can be misleading if the road sequence is interpreted directly from the raw kilometer values.

## Investment Value Adjustment

Investment values are annual nominal amounts. Before comparing investments across years, the project adjusts them for inflation using Brazil's official consumer price index, IPCA.

The inflation source is the IBGE SIDRA table 1737, using the variable "IPCA - Variação acumulada no ano" for December of each year. December is used because it represents the accumulated inflation for the full calendar year.

Investment values will be converted to a common price base before being compared with fatal crash metrics. This avoids comparing nominal amounts from different years as if they had the same purchasing power.

Source: https://apisidra.ibge.gov.br/values/t/1737/n1/1/v/69/p/201012,201112,201212,201312,201412,201512,201612,201712,201812,201912,202012,202112,202212,202312,202412

## Investment Analysis Methodology

The investment analysis does not rely only on a simple comparison between investment in year `t` and fatal crashes in year `t + 1`. That one-year lag is useful as an exploratory view, but it is too limited to describe roadway infrastructure effects by itself.

Roadway investments can have persistent effects. Pavement rehabilitation, new lanes, shoulders, safety devices, drainage, signage and geometry improvements may change the road risk level for several years. For this reason, Step 4 evaluates whether high-investment periods are associated with later changes in the fatal-crash baseline.

The analysis includes annual corrected investment, annual fatal crashes, combined charts, simple lag views, rolling or accumulated investment windows and phase-based comparisons around high-investment periods.

This methodological note is documented in `reports/02_investment_methodology_note.md`.

## Article Scope And Modeling Extension

The article produced from this repository focuses on the exploratory findings developed so far:

- changes in fatal-crash levels over time;
- dangerous kilometer zones using a corrected linear road reference;
- crash types and vehicle patterns associated with higher mortality;
- inflation-adjusted investment levels;
- investment phases and possible persistent roadway-safety effects;
- documentary evidence about roadway works and concession context.

The available investment series is annual and too limited to estimate reliably how much money would be required to reduce a specified number of fatal crashes. Therefore, the article treats investment findings as exploratory associations and methodological evidence, not as a causal prediction model.

After the article analysis, the repository may be extended with machine learning at occurrence level. That complementary work will investigate whether a crash is fatal from features such as crash type, vehicle involvement, corrected kilometer range and temporal or road characteristics. It is intentionally outside the core investment-versus-crashes article scope.

## Repository Structure

```text
.
|-- data/
|   |-- raw/
|   |-- processed/
|   `-- README.md
|-- analysis/
|   |-- 01_analise_inicial.py
|   |-- 02_hotspots_km_veiculos.py
|   |-- 03_investimentos.py
|   `-- 04_modelagem.py
|-- reports/
|   |-- figures/
|   |   |-- 01_analise_inicial/
|   |   |-- 02_hotspots/
|   |   |-- 03_investimentos/
|   |   `-- 04_modelos/
|   |--01_initial_report.md
|   |--02_investment_methodology_note.md
|   |--03_conclusion_after_phase4.md
|   `-- artigo/
|-- src/
|   |-- data_processing.py
|   |-- visualization.py
|   `-- modeling.py
|-- GUIA_NOVAS_CONVERSAS.md
|-- PLANO_DO_PROJETO.md
|-- requirements.txt
`-- .gitignore
```

## Known Issues and Limitations

- The project is still in progress, so conclusions are preliminary.
- The investment dataset is annual and aggregated by concessionaire; it does not show exactly where each investment was applied.
- A simple one-year lag may not capture persistent effects from infrastructure investment.
- The crash dataset has a kilometer discontinuity that required a corrected spatial reference.
- The year 2026 may be partial and should not be compared directly with complete years without adjustment.
- Fatal crashes are relatively rare compared with total crashes; any future machine learning extension must handle class imbalance carefully.
- Investment and fatal crash reduction must not be interpreted as causal without additional statistical evidence and domain validation.
- Traffic exposure data, such as AADT or vehicle-kilometers traveled, is not currently included.
- Investment values must be adjusted for inflation before historical comparison; nominal values alone can be misleading.

## Semantic Versioning

This repository uses semantic versioning while the project evolves:

- `v0.1.0-alpha`: initial project structure and raw data exploration.
- `v0.2.0-alpha`: initial charts completed and kilometer correction added.
- `v0.3.0-alpha`: hotspot analysis using corrected kilometer reference.
- `v0.4.0-alpha`: investment comparison, phase analysis and article preparation.
- `v0.5.0-alpha`: planned article completion and public analysis refinement.
- `v1.0.0`: planned first complete public exploratory analysis and article.
- Future extension: occurrence-level machine learning baseline.

## Next Step

The next development step is to consolidate the exploratory findings, figures, methodology and limitations into the technical article. Occurrence-level machine learning remains a future complementary extension of the repository.

*Developed by a Data Scientist / Analyst and Transport Engineer - Luciano Faria.*
