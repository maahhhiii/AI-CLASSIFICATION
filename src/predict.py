import joblib
import model

model = joblib.dump(model, "model/model.pkl")

def predict_species(features):
    prediction = model.predict([features])

    labels = {
        0: "Setosa",
        1: "Versicolor",
        2: "Virginica"
    }

    return labels[prediction[0]]