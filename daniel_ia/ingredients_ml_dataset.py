from datasets import load_dataset
import pandas as pd
import re

# Charger le dataset depuis Hugging Face
dataset = load_dataset("Thefoodprocessor/allergy_type")

# Convertir en DataFrame Pandas
df = pd.DataFrame(dataset["train"])

# Vérifier les colonnes disponibles
print("📊 Colonnes disponibles :", df.columns)

# Vérification des colonnes nécessaires
if not all(col in df.columns for col in ["recipe", "allergy_type"]):
    raise ValueError("❌ Erreur : Colonnes requises non trouvées dans le dataset.")

# Extraction des ingrédients depuis la colonne "recipe"
def extraire_ingredients(texte):
    """Extrait une liste d'ingrédients en utilisant un regex de base."""
    ingredients = re.findall(r"\b(?:[a-zA-Z\s-]+)\b", texte)  # Capture les mots
    ingredients = [i.strip().lower() for i in ingredients if len(i) > 2]  # Nettoyage
    return ", ".join(set(ingredients))  # Retourne une chaîne propre

df["ingredients"] = df["recipe"].apply(extraire_ingredients)

# Nettoyage de la colonne des allergènes
df["allergy_type"] = df["allergy_type"].str.lower().str.strip()

# Supprimer les lignes sans allergènes (pour conserver des données utiles au ML)
df = df.dropna(subset=["ingredients", "allergy_type"])

# Supprimer les doublons
df = df.drop_duplicates()

# Sauvegarde du fichier CSV
df[["ingredients", "allergy_type"]].to_csv("huggingface_recipes.csv", index=False, encoding="utf-8")
print("✅ Fichier généré : 'huggingface_recipes.csv'")
