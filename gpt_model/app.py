import openai
import pickle
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import MarianMTModel, MarianTokenizer  # Traduction locale

app = Flask(__name__)
CORS(app)  # Active le CORS pour permettre les requêtes depuis une app externe

# Charger la clé API OpenAI depuis key.txt
with open("key.txt", "r") as f:
    OPENAI_API_KEY = f.read().strip()

# Charger les modèles Machine Learning pour la prédiction de recettes
try:
    model = pickle.load(open('./model/model.pkl', 'rb'))
    tfidf = pickle.load(open('./model/tfidf.pkl', 'rb'))
    mlb = pickle.load(open('./model/mlb.pkl', 'rb'))
    print("✅ Modèle de prédiction chargé avec succès.")
except Exception as e:
    print(f"❌ Erreur lors du chargement du modèle de prédiction : {e}")

# Charger les modèles de traduction locale (français ↔ anglais)
try:
    model_fr_en = "Helsinki-NLP/opus-mt-fr-en"
    model_en_fr = "Helsinki-NLP/opus-mt-en-fr"

    translator_fr_en = MarianMTModel.from_pretrained(model_fr_en)
    tokenizer_fr_en = MarianTokenizer.from_pretrained(model_fr_en)

    translator_en_fr = MarianMTModel.from_pretrained(model_en_fr)
    tokenizer_en_fr = MarianTokenizer.from_pretrained(model_en_fr)

    print("✅ Modèles de traduction chargés avec succès.")
except Exception as e:
    print(f"❌ Erreur lors du chargement des modèles de traduction : {e}")

def translate_text_local(text, target_lang="EN"):
    """
    Traduit le texte vers la langue cible en utilisant MarianMT.
    """
    tokenizer = tokenizer_fr_en if target_lang == "EN" else tokenizer_en_fr
    model = translator_fr_en if target_lang == "EN" else translator_en_fr

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    translated_tokens = model.generate(**inputs)
    translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    return translated_text

def ask_chatgpt(food):
    """
    Envoie une requête à GPT-4o-Mini pour obtenir les allergènes confirmés et potentiels d'un aliment.
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

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": prompt}]
        )
        answer = response.choices[0].message.content.strip()
        return eval(answer) if answer.startswith("{") else {"allergens_confirmed": [], "allergens_potential": []}
    except Exception as e:
        print(f"⚠ Erreur GPT : {e}")
        return {"allergens_confirmed": ["Erreur"], "allergens_potential": []}

@app.route("/get_allergens", methods=["GET"])
def get_allergen_info():
    """
    Endpoint pour obtenir les allergènes d'un aliment via GPT-4o-Mini.
    """
    food = request.args.get("food", "").strip().lower()
    if not food:
        return jsonify({"error": "Aucun aliment fourni"}), 400

    allergens_data = ask_chatgpt(food)
    return jsonify(allergens_data)

@app.route("/predict_recipe", methods=["POST"])
def predict_recipe():
    """
    Endpoint pour obtenir les allergènes d'une recette complète avec le modèle local.
    Traduction automatique en anglais avant analyse et en français après analyse.
    """
    data = request.get_json()
    recipe_text = data.get("recipe", "").strip()

    if not recipe_text:
        return jsonify({"Predicted_Allergens": ["Erreur : Aucune recette fournie"]}), 400

    # Traduire la recette en anglais
    translated_recipe = translate_text_local(recipe_text, "EN")
    print(f"🔄 Recette traduite en anglais : {translated_recipe}")

    # Analyse par le modèle de Machine Learning
    try:
        recipe_vec = tfidf.transform([translated_recipe])
        prediction = model.predict(recipe_vec)
        predicted_allergies = mlb.inverse_transform(prediction)[0]

        # Traduire la réponse en français
        translated_allergens = [translate_text_local(a, "FR") for a in predicted_allergies]

    except Exception as e:
        print(f"⚠ Erreur de prédiction : {e}")
        return jsonify({"Predicted_Allergens": ["Erreur lors de la prédiction"]}), 500

    return jsonify({"Predicted_Allergens": translated_allergens})

if __name__ == "__main__":
    print("🚀 Serveur Flask démarré sur http://127.0.0.1:5000/")
    app.run(debug=True, host="0.0.0.0", port=5000)
