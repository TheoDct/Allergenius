import pandas as pd

def extraire_ingredients_allergenes(fichier_csv, min_occurrences=1):
    """
    Extrait les associations ingrédient ↔ allergène de la base de données.

    Args:
        fichier_csv (str): Nom du fichier CSV contenant les données prétraitées.
        min_occurrences (int): Seuil minimal d'occurrences pour garder une association.
    """
    # Charger les données prétraitées
    df = pd.read_csv(fichier_csv, low_memory=False, usecols=["ingredients_text", "allergens"])

    # Remplacer les valeurs NaN par des chaînes vides
    df = df.fillna("")

    # Séparer les ingrédients et allergènes en listes
    df["ingredients"] = df["ingredients_text"].str.split(",")
    df["allergens"] = df["allergens"].str.split(",")

    # Transformer chaque ingrédient et chaque allergène en une ligne distincte (évite les boucles !)
    df_exploded = df.explode("ingredients").explode("allergens")

    # Nettoyage : enlever les espaces inutiles et mettre en minuscule
    df_exploded["ingredients"] = df_exploded["ingredients"].str.strip().str.lower()
    df_exploded["allergens"] = df_exploded["allergens"].str.strip().str.lower()

    # Supprimer les lignes vides
    df_exploded = df_exploded[(df_exploded["ingredients"] != "") & (df_exploded["allergens"] != "")]

    # Compter les occurrences des associations ingrédient ↔ allergène
    df_counts = df_exploded.groupby(["ingredients", "allergens"]).size().reset_index(name="count")

    # Filtrer uniquement les associations ayant un nombre d'occurrences suffisant
    df_counts = df_counts[df_counts["count"] >= min_occurrences]

    # Trier les résultats
    df_counts = df_counts.sort_values(by=["count"], ascending=False)

    # Sauvegarder le fichier final
    df_counts.to_csv("ingredients_allergens.csv", index=False)

    print(f"✅ Extraction terminée : {len(df_counts)} associations enregistrées sous 'ingredients_allergens.csv'")
    return df_counts

# Exécution du script sur une base de données volumineuse
extraire_ingredients_allergenes("openfoodfacts_preprocessed.csv", min_occurrences=2)
