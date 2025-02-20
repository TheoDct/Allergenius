import pandas as pd
import json
import re
from tqdm import tqdm
from unidecode import unidecode
from nltk.stem import WordNetLemmatizer
import nltk

# Télécharger les ressources de lemmatisation NLTK
nltk.download('wordnet')

# Initialiser lemmatiser
lemmatizer = WordNetLemmatizer()

# Charger la liste des ingrédients Open Food Facts
with open("openfoodfacts_ingredients.json", "r", encoding="utf-8") as f:
    raw_ingredients = json.load(f)

# Normaliser et lemmatiser les ingrédients
ingredient_list = {lemmatizer.lemmatize(unidecode(ing.lower().strip())) for ing in raw_ingredients}

# Charger le dataset Hugging Face
csv_file = "huggingface_recipes_lemmatized.csv"
df = pd.read_csv(csv_file)

# Vérifier si la colonne "ingredients" existe
if "ingredients" not in df.columns:
    raise ValueError("❌ Erreur : La colonne 'ingredients' est absente du fichier CSV.")

# Nettoyage des valeurs NaN avant traitement
df = df.dropna(subset=["ingredients"])

# Fonction améliorée pour extraire les ingrédients
def extract_ingredients_advanced(recipe_text):
    words = set(lemmatizer.lemmatize(unidecode(recipe_text.lower())).split())  # Normalisation + tokenisation + lemmatisation
    found_ingredients = [ingredient for ingredient in ingredient_list if ingredient in words]  # Recherche par intersection rapide
    return found_ingredients

print("🔄 Début du traitement des recettes...")

# Appliquer la fonction avec une barre de progression
df["matched_ingredients"] = [
    extract_ingredients_advanced(text) for text in tqdm(df["ingredients"], desc="📊 Traitement des recettes")
]

# Sauvegarde des résultats
output_file = "recipes_with_ingredients_v2.csv"
df.to_csv(output_file, index=False, encoding="utf-8")

print(f"✅ Extraction terminée. Résultats enregistrés dans '{output_file}'.")
