import pandas as pd
from deep_translator import GoogleTranslator

def pretraiter_donnees(fichier_csv):
    # Charger les données
    df = pd.read_csv(fichier_csv, low_memory=False)

    # Sélectionner uniquement les colonnes importantes
    colonnes_utiles = ["product_name", "ingredients_text", "allergens", "brands", "countries"]
    df = df[colonnes_utiles].copy()

    # Suppression des valeurs nulles
    df.dropna(subset=["ingredients_text"], inplace=True)

    # Convertir tout en minuscules pour uniformiser
    df["ingredients_text"] = df["ingredients_text"].str.lower()
    df["allergens"] = df["allergens"].str.lower()

    # Nettoyer les caractères spéciaux des ingrédients
    df["ingredients_text"] = df["ingredients_text"].str.replace(r"[^\w\s,]", "", regex=True)

    # Supprimer les doublons
    df.drop_duplicates(inplace=True)

    # Traduire les ingrédients en anglais si nécessaire
    def traduire_ingredients(ingredients):
        if any(char.isalpha() and char not in "abcdefghijklmnopqrstuvwxyz " for char in ingredients.lower()):
            return GoogleTranslator(source='auto', target='en').translate(ingredients)
        return ingredients

    df["ingredients_text"] = df["ingredients_text"].apply(traduire_ingredients)

    # Nettoyage des allergènes (suppression de "en:" et séparation propre)
    df["allergens"] = df["allergens"].apply(lambda x: x.replace("en:", "").replace(",", ", ") if pd.notna(x) else "No allergens detected")

    # Sauvegarde du fichier nettoyé
    df.to_csv("openfoodfacts_preprocessed.csv", index=False)
    print("✅ Données prétraitées et enregistrées sous 'openfoodfacts_preprocessed.csv'")

# Exécuter le prétraitement
pretraiter_donnees("openfoodfacts_sample.csv")
