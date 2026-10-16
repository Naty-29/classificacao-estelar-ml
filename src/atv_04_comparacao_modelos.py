import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


df = pd.read_csv('data/processed/star_classification_tratado.csv')

print(df.head())
print(df.shape) # Usar quando quiser linhas e colunas
print(df.info()) # Fornece os tipos de cada coluna
print(df.describe())
print(df.isnull().sum()) # Nulos por coluna
print(df.duplicated().sum())

mascara_outlier = (df[['u', 'g', 'z']] < -1000).any(axis = 1)
print(f"Quantidade de ouliers: {mascara_outlier.sum()}")

df = df[~mascara_outlier].reset_index(drop=True)
df.to_csv('star_classification_tratado.csv', index=False)
print(f"Formato depois da remoção: {df.shape}")

id_cols = ['obj_ID', 'run_ID', 'rerun_ID', 'cam_col', 'field_ID', 'spec_obj_ID', 'plate', 'MJD', 'fiber_ID']

df_modelo =  df.drop(columns = id_cols)

print(df_modelo.shape)
print(df_modelo.columns.tolist())

# PRÉ PROCESSAMENTO

# Separar X (atributos) e y (alvo)
X = df_modelo.drop(columns = ['class'])
y_texto = df_modelo['class']

# Codificar o nosso alvo
le = LabelEncoder()
y = le.fit_transform(y_texto) # Troca o texto por números
print(dict(zip(le.classes_, range(len(le.classes_))))) # Vai trazer qual número corresponde a qual classe


# COMPARAÇÃO DE MODELOS
# KNN

# COMPARAÇÃO DE MODELOS (5-FOLD)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
metricas = {'accuracy': 'accuracy', 'f1_macro': 'f1_macro'}


def avaliar(modelo, nome):
    res = cross_validate(modelo, X, y, cv=cv, scoring=metricas, n_jobs=-1)
    acc = res['test_accuracy']
    f1 = res['test_f1_macro']
    print(f"{nome}: acurácia = {acc.mean():.4f} (+/- {acc.std():.4f}) | "
          f"F1-macro = {f1.mean():.4f}")
    return {'acuracia': acc.mean(), 'f1_macro': f1.mean()}


resultados = {}

# Árvore de Decisão
print ("\nDecision Tree")
resultados['Decision Tree'] = avaliar(
    DecisionTreeClassifier(random_state=42), 'Decision Tree')

# Random Forest
print("\nRandom Forest")
resultados['Random Forest'] = avaliar(
    # parâmetro n_jobs=-1 nos permite usar todos os núcleos do processador, o que acelera a execução
    RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Random Forest')

# KNN (testa vários valores de K) 
print("\nKNN")
k_values = [1, 3, 5, 7, 9, 11, 15, 21]
resultados_knn = {}

for k in k_values:
    modelo_knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))
    resultados_knn[k] = avaliar(modelo_knn, f'KNN (K={k})')

melhor_k = max(resultados_knn, key=lambda k: resultados_knn[k]['acuracia'])
print(f"Melhor K: {melhor_k} (acurácia = {resultados_knn[melhor_k]['acuracia']:.4f})")
resultados[f'KNN (K={melhor_k})'] = resultados_knn[melhor_k]

# Naive Bayes 
print("\nNaive Bayes")
resultados['Naive Bayes'] = avaliar(
    make_pipeline(StandardScaler(), GaussianNB()), 'Naive Bayes')

# Comparação dos modelos
# RESULTADO FINAL
tabela = pd.DataFrame(resultados).T.sort_values('acuracia', ascending=False)

print("\n ------- RESULTADO FINAL (5-fold, ordenado por acurácia) -------")
print(tabela.round(4))

melhor_modelo = tabela.index[0]
print(f"\nMelhor algoritmo: {melhor_modelo}")
print(f"Acurácia média: {tabela.loc[melhor_modelo, 'acuracia']:.4f}")
print(f"F1-macro médio: {tabela.loc[melhor_modelo, 'f1_macro']:.4f}")