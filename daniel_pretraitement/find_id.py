import pandas as pd

def afficher_produit_par_id(fichier_csv, id_produit):
    # Charger le fichier CSV
    df = pd.read_csv(fichier_csv, low_memory=False)
    
    # Vérifier si l'ID est valide
    if id_produit < 0 or id_produit >= len(df):
        print("❌ ID invalide. Veuillez entrer un nombre entre 0 et", len(df) - 1)
        return
    
    # Sélectionner uniquement les colonnes importantes
    colonnes_utiles = ["product_name", "ingredients_text", "allergens", "brands", "countries"]
    produit = df.iloc[id_produit][colonnes_utiles]

    # Afficher les résultats de manière propre
    print("\n🔎 **Détails du produit (ID:", id_produit, ")**")
    print(f"🛒 **Nom du produit :** {produit['product_name']}")
    print(f"🥦 **Ingrédients :** {produit['ingredients_text']}")
    print(f"⚠️ **Allergènes :** {produit['allergens'] if pd.notna(produit['allergens']) else 'Aucun'}")

# Exemple d'utilisation
fichier_csv = "openfoodfacts_preprocessed.csv"  # Remplace par le bon chemin si nécessaire
id_produit = int(input("🔢 Entrez l'ID du produit : "))
afficher_produit_par_id(fichier_csv, id_produit)
