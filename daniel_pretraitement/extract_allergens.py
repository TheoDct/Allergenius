import pandas as pd 

def extraire_allergenes(fichier_csv):
    # Charger les données nettoyées
    df = pd.read_csv(fichier_csv, low_memory=False)

    # Vérifier si la colonne "allergens" existe
    if "allergens" not in df.columns:
        print("❌ Erreur : La colonne 'allergens' n'existe pas dans le fichier.")
        return

    # Extraire et nettoyer les allergènes
    allergenes = df["allergens"].dropna().str.split(",")

    # Aplatir la liste et supprimer les espaces inutiles
    allergenes_uniques = set(allergene.strip().replace("en:", "").replace("fr:", "") 
                             for liste in allergenes for allergene in liste if allergene.strip())

    # Supprimer "No allergens detected" s'il existe
    allergenes_uniques.discard("No allergens detected")

    # Convertir en DataFrame et sauvegarder
    df_allergenes = pd.DataFrame(sorted(allergenes_uniques), columns=["allergen"])
    df_allergenes.to_csv("allergens_list_cleaned.csv", index=False)

    print("✅ Liste des allergènes nettoyée et enregistrée sous 'allergens_list_cleaned.csv'")
    return df_allergenes

# Exemple d'utilisation
extraire_allergenes("openfoodfacts_preprocessed.csv")
