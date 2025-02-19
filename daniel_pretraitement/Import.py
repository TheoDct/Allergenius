import requests
import pandas as pd

# URL de l'API Open Food Facts
API_URL = "https://world.openfoodfacts.org/api/v2/search"

# Paramètres pour récupérer les données
params = {
    "fields": "product_name,ingredients_text,allergens,brands,countries",
    "page_size": 500,  # Nombre de produits à récupérer
    "json": 1
}

# Requête à l'API
response = requests.get(API_URL, params=params)

# Vérifier si la requête a réussi
if response.status_code == 200:
    data = response.json()
    products = data.get("products", [])

    # Convertir en DataFrame
    df_openfood = pd.DataFrame(products)

    # Afficher un aperçu des données
    print(df_openfood.head())

    # Sauvegarder en CSV
    df_openfood.to_csv("openfoodfacts_sample.csv", index=False)
    print("✅ Données enregistrées sous 'openfoodfacts_sample.csv'")

else:
    print("❌ Erreur lors de la récupération des données :", response.status_code)
