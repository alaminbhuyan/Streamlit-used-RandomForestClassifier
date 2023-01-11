import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd

def draw_meshgrid():
    a = np.arange(start=X[:, 0].min() - 1, stop=X[:, 0].max() + 1, step=0.01)
    b = np.arange(start=X[:, 1].min() - 1, stop=X[:, 1].max() + 1, step=0.01)

    XX, YY = np.meshgrid(a, b)

    input_array = np.array([XX.ravel(), YY.ravel()]).T

    return XX, YY, input_array

df = pd.read_csv('concertriccir2.csv')
X = df.iloc[:,0:2].values
y = df.iloc[:,-1].values


# Split the dataset into train test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

# Set style
plt.style.use(style="fivethirtyeight")

st.sidebar.markdown(body="RandomForest Classifier")

n_estimators = int(st.sidebar.number_input(label="N-estimators"))

max_features = st.sidebar.selectbox(
    label="Max Features",
    options=('auto', 'sqrt','log2','manual')
)

if max_features == "manual":
    max_features = int(st.sidebar.number_input(label="Max Features"))

bootstrap = st.sidebar.selectbox(label="Bootstrap", options=(True, False))

max_samples = st.sidebar.slider(label="Max Samples", min_value=1, max_value=X_train.shape[0], value=1, key="1236")

# Load initial graph
fig, ax = plt.subplots()

# Plot initial graph
ax.scatter(X.T[0], X.T[1], c=y, cmap='rainbow')
origin = st.pyplot(fig=fig)

if st.sidebar.button(label="Run Algorithm"):
    origin.empty()

    if n_estimators == 0:
        n_estimators = 100

    # n_estimators = Number of Decision Tree
    # max_features = Apply Column sampling (number of features or columns)
    # max_samples = Apply row sampling (number of rows)
    # bootstrap = If True then we we apply row or column sampling then it will take value randomly
    rfc = RandomForestClassifier(n_estimators=n_estimators, random_state=42, bootstrap=bootstrap,
                                 max_samples=max_samples, max_features=max_features)
    rfc.fit(X=X_train, y=y_train)
    y_pred = rfc.predict(X=X_test)

    XX, YY, input_array = draw_meshgrid()

    labels = rfc.predict(X=input_array)

    ax.contour(XX, YY, labels.reshape(XX.shape), alpha=0.5, cmap="rainbow")
    plt.xlabel(xlabel="Col1")
    plt.ylabel(ylabel="Col2")
    origin = st.pyplot(fig=fig)
    st.header(body="Accuracy: " + str(round(accuracy_score(y_true=y_test, y_pred=y_pred), 2)))