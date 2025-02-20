import pandas as pd
import re

def pretraiter_donnees(fichier_csv):
    # Charger les données brutes
    df = pd.read_csv(fichier_csv, low_memory=False)

    # Vérifier la présence des colonnes nécessaires
    colonnes_requises = ["product_name", "ingredients_text", "allergens", "brands", "countries"]
    if not all(col in df.columns for col in colonnes_requises):
        print("❌ Erreur : Colonnes essentielles manquantes.")
        return

    # Convertir en minuscules pour uniformiser
    df["ingredients_text"] = df["ingredients_text"].astype(str).str.lower()
    df["allergens"] = df["allergens"].astype(str).str.lower()

    # Nettoyer les caractères spéciaux dans les ingrédients et supprimer les chiffres inutiles
    df["ingredients_text"] = df["ingredients_text"].apply(lambda x: re.sub(r"[^\w\s,]", "", x))  # Supprimer caractères spéciaux
    df["ingredients_text"] = df["ingredients_text"].apply(lambda x: re.sub(r"\d+(\.\d+)?", "", x).strip())  # Supprimer chiffres

    # Nettoyage des allergènes (suppression des préfixes type "en:")
    df["allergens"] = df["allergens"].apply(lambda x: re.sub(r"[a-z]{2}:", "", x))

    # Sauvegarder les données nettoyées
    df.to_csv("openfoodfacts_preprocessed.csv", index=False)
    print("✅ Données prétraitées enregistrées sous 'openfoodfacts_preprocessed.csv'")

    return df

# Exécution du script
pretraiter_donnees("openfoodfacts_sample.csv")
