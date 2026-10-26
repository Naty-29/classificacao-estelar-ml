# Classificação Estelar com Machine Learning

Projeto de classificação de objetos astronômicos em **GALAXY**, **STAR** e **QSO** (quasar)
a partir de dados do Sloan Digital Sky Survey (SDSS), usando Python e scikit-learn.

O trabalho tem duas etapas: a comparação de quatro algoritmos de classificação e a
otimização de hiperparâmetros de um SVM com `GridSearchCV`.

## Dataset

- **Fonte:** [Stellar Classification Dataset - SDSS17](https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17) (Kaggle, autor: fedesoriano)
- **Tamanho usado:** 99.999 observações. A base original do Kaggle tem 100.000 registros, e o arquivo em `data/raw/` já não contém o registro com valor `-9999` em `u`, `g` e `z` (outlier conhecido da base).
- **Alvo:** coluna `class` (GALAXY, QSO ou STAR)
- **Atributos utilizados:** `alpha`, `delta`, `u`, `g`, `r`, `i`, `z` (filtros fotométricos) e `redshift`
- **Atributos descartados:** colunas identificadoras, que não carregam informação física do objeto (`obj_ID`, `run_ID`, `rerun_ID`, `cam_col`, `field_ID`, `spec_obj_ID`, `plate`, `MJD`, `fiber_ID`)

## Estrutura do repositório

```
classificacao-estelar-ml/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   │   └── star_classification.csv
│   └── processed/
│       └── star_classification_tratado.csv
└── src/
    ├── atv_04_comparacao_modelos.py
    └── atv_05_otimizacao_svm.py
```

## Etapas

### 1. Análise, tratamento e comparação de modelos (`src/atv_04_comparacao_modelos.py`)

- Análise exploratória: formato, tipos, valores nulos, duplicados e estatísticas descritivas
- Remoção de outliers (valores abaixo de -1000 em `u`, `g` ou `z`) e geração do CSV tratado
- Remoção das colunas identificadoras
- Codificação do alvo com `LabelEncoder`
- Comparação com **validação cruzada estratificada (5-fold)** de:
  - Decision Tree
  - Random Forest (100 árvores)
  - KNN (K = 1, 3, 5, 7, 9, 11, 15 e 21), com `StandardScaler`
  - Naive Bayes Gaussiano, com `StandardScaler`
- Métricas: acurácia e F1-macro

### 2. Otimização de hiperparâmetros com SVM (`src/atv_05_otimizacao_svm.py`)

- Amostra estratificada de 5.000 registros
- Separação treino/teste (80/20), estratificada
- Pipeline com `StandardScaler` + `SVC`
- `GridSearchCV` com `StratifiedKFold` (5 folds), variando:
  - `C`: 0,01, 0,1, 1, 10 e 100
  - `kernel`: linear, poly, rbf e sigmoid

## Resultados

### Comparação de modelos (validação cruzada estratificada, 5-fold)

| Modelo | Acurácia | F1-macro |
|--------|----------|----------|
| **Random Forest** | **0,9780** | **0,9744** |
| Decision Tree | 0,9652 | 0,9599 |
| KNN (K=3) | 0,9395 | 0,9330 |
| Naive Bayes | 0,9221 | 0,9134 |

O **Random Forest** teve o melhor desempenho. No KNN, o melhor valor de K foi 3, e a
acurácia diminuiu de forma gradual conforme K aumentou.

### Otimização do SVM (GridSearchCV, 5-fold, amostra de 5.000 registros)

Foram testadas 20 combinações (5 valores de `C` × 4 kernels), totalizando 100 ajustes.

| Posição | Kernel | C | Acurácia (CV) |
|---------|--------|---|---------------|
| 1º | linear | 100 | 0,9635 |
| 2º | linear | 10 | 0,9600 |
| 3º | linear | 1 | 0,9547 |
| 4º | rbf | 10 | 0,9495 |
| 5º | rbf | 100 | 0,9420 |

- **Melhores parâmetros:** `kernel=linear`, `C=100`
- **Acurácia média na validação cruzada:** 0,9635
- **Acurácia no conjunto de teste (20% da amostra):** 0,9710

O kernel linear superou os demais, o que sugere que as classes são bem separáveis com os
atributos usados. Com o kernel linear, a acurácia cresceu a cada aumento de `C`, e o melhor
valor está no limite superior da grade, então valores maiores de `C` poderiam ser testados.

### Conclusão

O Random Forest (97,80%) superou o melhor SVM (96,35% na validação cruzada). Essa comparação
exige cautela: o SVM foi treinado com apenas 5.000 registros, enquanto os demais modelos
usaram a base completa.

## Como executar

Execute os comandos a partir da raiz do projeto.

```bash
git clone https://github.com/Naty-29/classificacao-estelar-ml.git
cd classificacao-estelar-ml
pip install -r requirements.txt

python src/atv_04_comparacao_modelos.py   # análise, tratamento e comparação de modelos
python src/atv_05_otimizacao_svm.py       # otimização do SVM
```

O primeiro script gera o arquivo `star_classification_tratado.csv`, que é usado pelo segundo.


## Tecnologias

Python · pandas · scikit-learn

## Autora

Natália Santos · [GitHub](https://github.com/Naty-29) · [LinkedIn](https://www.linkedin.com/in/natália-santos-670a58259)
