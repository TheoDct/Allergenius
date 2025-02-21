import openai
import pickle
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Autorise l'accès depuis une app Node.js

# Charger la clé API depuis key.txt
with open("key.txt", "r") as f:
    OPENAI_API_KEY = f.read().strip()

# Charger les modèles Machine Learning pour la prédiction de recettes
model = pickle.load(open('./model/model.pkl', 'rb'))
tfidf = pickle.load(open('./model/tfidf.pkl', 'rb'))
mlb = pickle.load(open('./model/mlb.pkl', 'rb'))

def ask_chatgpt(food):
    """
    Envoie une requête à ChatGPT (GPT-4o-Mini) pour obtenir les allergènes confirmés et potentiels de l'aliment.
    """
    client = openai.OpenAI(api_key=OPENAI_API_KEY)

    prompt = f"""
    Tu es un expert en nutrition et allergies alimentaires.
    Analyse l'aliment suivant : "{food}".
    
    1. **Allergènes confirmés** : Liste les allergènes naturellement présents dans cet aliment.
    2. **Allergènes potentiels** : Liste les allergènes pouvant être présents en raison d'une contamination croisée ou transformation industrielle.

    Réponds strictement sous ce format JSON :
    {{
        "allergens_confirmed": ["allergène1", "allergène2"],
        "allergens_potential": ["allergène3", "allergène4"]
    }}

    Exemples :
    - "pain" -> {{"allergens_confirmed": ["gluten"], "allergens_potential": ["soja", "sésame"]}}
    - "purée de carotte" -> {{"allergens_confirmed": [], "allergens_potential": ["céleri"]}}
    - "beurre" -> {{"allergens_confirmed": ["lait"], "allergens_potential": []}}

    Réponds uniquement avec le JSON, sans texte supplémentaire.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # ⚡ Utilisation de GPT-4o-Mini
        messages=[{"role": "system", "content": prompt}]
    )

    try:
        answer = response.choices[0].message.content.strip()
        data = eval(answer) if answer.startswith("{") else {"allergens_confirmed": [], "allergens_potential": []}
        return data
    except:
        return {"allergens_confirmed": ["Erreur de récupération"], "allergens_potential": []}

@app.route("/get_allergens", methods=["GET"])
def get_allergen_info():
    food = request.args.get("food", "").strip().lower()
    allergens_data = ask_chatgpt(food)
    return jsonify(allergens_data)

@app.route("/predict_recipe", methods=["POST"])
def predict_recipe():
    """
    Utilise le modèle local (pickle) pour prédire les allergènes d'une recette complète.
    """
    data = request.get_json()
    recipe_text = data.get("recipe", "").strip()

    if not recipe_text:
        return jsonify({"Predicted_Allergens": ["Erreur : Aucune recette fournie"]})

    recipe_vec = tfidf.transform([recipe_text])
    prediction = model.predict(recipe_vec)
    predicted_allergies = mlb.inverse_transform(prediction)[0]

    return jsonify({"Predicted_Allergens": list(predicted_allergies)})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
