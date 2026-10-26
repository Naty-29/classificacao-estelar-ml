import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

df = pd.read_csv('data/processed/star_classification_tratado.csv')

# Colunas identificadoras: não carregam informação física do objeto
id_cols = ['obj_ID', 'run_ID', 'rerun_ID', 'cam_col', 'field_ID',
           'spec_obj_ID', 'plate', 'MJD', 'fiber_ID']
df = df.drop(columns=id_cols)

COLUNA_ALVO = 'class'
AMOSTRA = 5000

 
# Amostra estratificada
if AMOSTRA and len(df) > AMOSTRA:
    df, _ = train_test_split(
        df, train_size=AMOSTRA, random_state=42, stratify=df[COLUNA_ALVO]
    )
 
X = df.drop(columns=[COLUNA_ALVO])
y = df[COLUNA_ALVO]
 


# Separa uma parte para teste final
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#Pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC()),
])

param_grid = {
    "svc__C": [0.01, 0.1, 1, 10, 100],
    "svc__kernel": ["linear", "poly", "rbf", "sigmoid"],
}

# K-FOLD (5) 
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=kfold,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1,
)

grid.fit(X_train, y_train)

# RESULTADOS
print("\n--- RESULTADOS ---")
print("Melhores parâmetros:", grid.best_params_)
print(f"Melhor acurácia (média nos 5 folds): {grid.best_score_:.4f}")

y_pred = grid.best_estimator_.predict(X_test)
print(f"Acurácia no conjunto de teste: {accuracy_score(y_test, y_pred):.4f}")

# Ranking de todas as combinações que foram testadas
print("\n--- Top 5 combinações ---")
resultados = sorted(
    zip(grid.cv_results_["mean_test_score"], grid.cv_results_["params"]),
    key=lambda t: t[0],
    reverse=True,
)
for score, params in resultados[:5]:
    print(f"{score:.4f}  ->  {params}")