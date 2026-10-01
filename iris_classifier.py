#Implementação de um classificador de flores iris usando o algoritmo KNN com scikit-learn e a biblioteca Streamlit 

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# dataset de flores iris contendo 150 amostras, 4 features e 3 classes
from sklearn.datasets import load_iris #

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


st.set_page_config(
    page_title="Iris Explorer",
    layout="wide"
)

st.title("Iris Explorer")
st.write(
    "Explore o algoritmo KNN e descubra "
    "como ele classifica diferentes espécies de flores."
)
st.image("img\iris_info.jpg", caption="Legenda da imagem")


# 2. Carregar os dados

iris = load_iris()

X = iris.data
y = iris.target

df = pd.DataFrame(
    X,
    columns=iris.feature_names
)

df["especie"] = [
    iris.target_names[i] for i in y
]



# 3. Dividir os dados
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)



# 4. Controles da interface

st.sidebar.header("Configurações do KNN")

k = st.sidebar.slider(
    "Número de vizinhos (K)",
    min_value=1,
    max_value=15,
    value=5,
    step=1
)

st.sidebar.subheader("Medidas da flor")

sepal_length = st.sidebar.slider(
    "Comprimento da sépala (cm)",
    4.0, 8.0, 5.1, 0.1
)

sepal_width = st.sidebar.slider(
    "Largura da sépala (cm)",
    2.0, 4.5, 3.5, 0.1
)

petal_length = st.sidebar.number_input(
    "Comprimento da pétala (cm)",
    1.0, 7.0, 1.4, 0.1
)

petal_width = st.sidebar.slider(
    "Largura da pétala (cm)",
    0.1, 2.5, 0.2, 0.1
)



# 5. Treinar o modelo


modelo = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=k)
)

modelo.fit(X_train, y_train)


# 6. Fazer a previsão

nova_flor = [[
    sepal_length,
    sepal_width,
    petal_length,
    petal_width
]]

previsao = modelo.predict(nova_flor)[0]

probabilidades = modelo.predict_proba(nova_flor)[0]

nome_especie = iris.target_names[previsao]


# -------------------------
# 7. Mostrar resultados
# -------------------------

col1, col2 = st.columns(2)

with col1:
    st.subheader("Resultado da classificação")

    st.success(f"Espécie prevista: {nome_especie}")

    st.metric(
        "Votos dos vizinhos para a classe prevista",
        f"{probabilidades[previsao]:.1%}"
    )

with col2:
    st.subheader("Desempenho do modelo")

    y_pred = modelo.predict(X_test)

    acuracia = accuracy_score(y_test, y_pred)

    st.metric(
        "Acurácia no conjunto de teste",
        f"{acuracia:.2%}"
    )


# -------------------------
# 8. Encontrar os vizinhos
# -------------------------

st.subheader("Os K vizinhos mais próximos")

scaler = modelo.named_steps["standardscaler"]
knn = modelo.named_steps["kneighborsclassifier"]

X_train_scaled = scaler.transform(X_train)
nova_flor_scaled = scaler.transform(nova_flor)

distancias, indices = knn.kneighbors(
    nova_flor_scaled
)

vizinhos = X_train[indices[0]]

tabela_vizinhos = pd.DataFrame(
    vizinhos,
    columns=iris.feature_names
)

tabela_vizinhos["especie"] = [
    iris.target_names[y_train[i]]
    for i in indices[0]
]

tabela_vizinhos["distancia"] = distancias[0]

st.dataframe(
    tabela_vizinhos,
    use_container_width=True
)


# -------------------------
# 9. Gráfico de dispersão
# -------------------------

st.subheader("Visualização das espécies")

fig, ax = plt.subplots(figsize=(7, 4))

cores = {
    "setosa": "royalblue",
    "versicolor": "darkorange",
    "virginica": "seagreen"
}

for especie in iris.target_names:
    grupo = df[df["especie"] == especie]

    ax.scatter(
        grupo["petal length (cm)"],
        grupo["petal width (cm)"],
        label=especie,
        alpha=0.65,
        color=cores[especie]
    )

ax.scatter(
    petal_length,
    petal_width,
    color="red",
    marker=".",
    s=300,
    edgecolor="black",
    label="Nova flor"
)

ax.set_xlabel("Comprimento da pétala (cm)")
ax.set_ylabel("Largura da pétala (cm)")
ax.set_title("Distribuição das espécies de Iris")
ax.legend()
ax.grid(alpha=0.2)

st.pyplot(fig)