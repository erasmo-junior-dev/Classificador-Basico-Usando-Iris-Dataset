from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

# Carregando o Iris Dataset
iris_ds = datasets.load_iris()

# Carregando os dados do Dataset no Pandas
ds = pd.DataFrame(iris_ds.data, columns=iris_ds.feature_names)
ds['species'] = pd.Categorical.from_codes(iris_ds.target, iris_ds.target_names)

# Resumo do Dataset com aproximação de 2 casas decimais
print(ds.describe().round(2))

# Histograma da frequência do tamanho das petalas
histogram = ds['petal length (cm)'].plot.hist()
plt.title("Distribuição do Comprimento das Pétalas")
plt.xlabel("Petal Length (cm)")
plt.show()

# Gráfico de dispersão entre o comprimento da sepala e o comprimento das petalas
dispersion = ds.plot.scatter(x='petal length (cm)', y='sepal length (cm)')
plt.title("Dispersão Entre o Comprimento das Pétalas e Sépalas")
plt.xlabel('Petal Length (cm)')
plt.ylabel('Sepal Length (cm)')
plt.show()

# Gráfico de dispersão entre todos os dados
sb.pairplot(ds, hue='species')
plt.show()