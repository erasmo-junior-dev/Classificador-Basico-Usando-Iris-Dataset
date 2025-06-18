from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

# Carregando o Iris Dataset
iris_ds = datasets.load_iris()

# Carregando os dados do Dataset no Pandas
ds = pd.DataFrame(iris_ds.data, columns=iris_ds.feature_names)

# Resumo do Dataset com aproximação de 2 casas decimais
print(ds.describe().round(2))

