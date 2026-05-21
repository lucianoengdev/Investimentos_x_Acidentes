# BR-381 Fatal Crash Analysis

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge)
![scikit--learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)

**Current version:** `v0.3.0-alpha`

## What This Does

This project analyzes fatal crashes on the BR-381 / Autopista Fernao Dias highway and investigates how historical roadway investments may relate to fatal crash reduction.

Working features:

- Loads and cleans traffic crash data.
- Preserves the original kilometer marker and creates a corrected linear kilometer reference.
- Generates initial exploratory charts for fatal crashes, deaths, crash types and kilometer distribution.
- Organizes the project as an open-source data analysis repository.
- Prepares the foundation for later investment comparison and machine learning modeling.

## What Is This?

This is a **Python data analysis and data science project**, not a web, desktop or mobile application.

The code is expected to produce:

- cleaned datasets in `data/processed/`;
- exploratory charts in `reports/figures/`;
- documented analysis scripts in `analysis/`;
- reusable data-processing and visualization functions in `src/`;
- a future technical article about fatal crashes, investment and roadway safety.

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
- **scikit-learn**: planned machine learning models.
- **Git / GitHub Desktop**: version control workflow.
- **VS Code**: development environment.

## Project Ambition

The goal is to build a complete transportation data project. The project is not intended to simply produce charts. It is intended to document a full reasoning process:

1. understand the raw data;
2. clean and validate the data;
3. generate exploratory charts;
4. interpret the results before modeling;
5. compare crash patterns with roadway investment;
6. test predictive models carefully;
7. write a public technical article with conclusions and limitations.

The long-term ambition is to estimate whether investment levels appear related to fatal crash reduction, while being careful not to claim causality without enough evidence.

## Project Stage

As of May 21, 2026, this project is a **work in progress**.

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

### In Progress

- [ ] Step 4: Add investment data from `investimentos.csv`.

### Pending

- [ ] Compare investment with fatal crashes using time lag.
- [ ] Discuss whether investment can support exploratory cost-reduction scenarios.
- [ ] Step 5: Define the modeling question.
- [ ] Step 6: Train and evaluate machine learning models.
- [ ] Step 7: Write the final README updates and technical article.

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
|   |--01_initial_report.md/
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
- The crash dataset has a kilometer discontinuity that required a corrected spatial reference.
- The year 2026 may be partial and should not be compared directly with complete years without adjustment.
- Fatal crashes are relatively rare compared with total crashes, so future machine learning models must handle class imbalance carefully.
- Investment and fatal crash reduction must not be interpreted as causal without additional statistical evidence and domain validation.
- Traffic exposure data, such as AADT or vehicle-kilometers traveled, is not currently included.

## Semantic Versioning

This repository uses semantic versioning while the project evolves:

- `v0.1.0-alpha`: initial project structure and raw data exploration.
- `v0.2.0-alpha`: initial charts completed and kilometer correction added.
- `v0.3.0-alpha`: planned hotspot analysis using corrected kilometer reference.
- `v0.4.0-alpha`: planned investment comparison.
- `v0.5.0-alpha`: planned machine learning baseline.
- `v1.0.0`: planned first complete public analysis and article.

## Next Step

The next development step is to Analyze hotspots by corrected kilometer, road direction and vehicle type and write interpretations about it.

*Developed by a Data Scientist / Analyst and Transport Engineer - Luciano Faria.*