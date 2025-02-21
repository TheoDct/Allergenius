import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pickle
import pandas as pd

# fichier non utilisable en l'état (problème de compatibilité avec scikit-learn 1.6)

model = pickle.load(open('../model/model.pkl', 'rb'))
tfidf = pickle.load(open('../model/tfidf.pkl', 'rb'))
mlb = pickle.load(open('../model/mlb.pkl', 'rb'))  # MultiLabelBinarizer pour convertir les prédictions

# Initialisation de l'API
app = FastAPI()

# Définition du format de la requête attendue
# class RecipeInput(BaseModel):
#     recipe: str  # La recette sous forme de texte

# Fonction pour nettoyer le texte 
def clean_text(text):
    return text.lower().strip()

# Endpoint pour prédire les allergies à partir d'une recette
@app.post("/predict/")
def predict_recipe(data):  # Suppression de l'annotation RecipeInput
    try:
        # Nettoyage du texte
        recipe_text = clean_text(data["recipe"])  # Adaptation pour dictionnaire

        # Transformation en vecteur TF-IDF
        recipe_vec = tfidf.transform([recipe_text])

        # Prédiction avec le modèle
        prediction = model.predict(recipe_vec)

        # Conversion en liste de labels lisibles
        predicted_allergies = mlb.inverse_transform(prediction)[0]

        # Réponse JSON
        return {"Predicted_Allergens": predicted_allergies}



    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Lancement du serveur FastAPI avec Uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Commande pour lancer l'API : uvicorn API:app --reload