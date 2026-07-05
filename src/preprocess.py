import pandas as pd

def load_data():
    from sklearn.datasets import load_iris

    iris = load_iris()

    df = pd.DataFrame(
        iris.data,
        columns=iris.feature_names
    )

    df["species"] = iris.target

    return df