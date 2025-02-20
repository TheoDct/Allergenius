import requests
import json

URL = "https://world.openfoodfacts.org/ingredients.json"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

print("🔄 Récupération des ingrédients depuis OpenFoodFacts...")

try:
    response = requests.get(URL, headers=HEADERS, timeout=30)  # ⏳ Ajout d'un User-Agent + timeout
    response.raise_for_status()  # 🚨 Vérifie les erreurs HTTP

    # ✅ Vérifie que la réponse contient bien du JSON
    try:
        data = response.json()
    except json.JSONDecodeError:
        print("❌ La réponse de l'API OpenFoodFacts n'est pas du JSON valide !")
        print("🔍 Contenu reçu :", response.text[:500])  # Affiche les 500 premiers caractères pour debug
        exit(1)

    # Extraire uniquement la liste des ingrédients
    ingredients = [entry["name"] for entry in data.get("tags", []) if "name" in entry]

    # Sauvegarde dans un fichier JSON
    with open("openfoodfacts_ingredients.json", "w", encoding="utf-8") as f:
        json.dump(ingredients, f, indent=4, ensure_ascii=False)

    print(f"✅ {len(ingredients)} ingrédients enregistrés dans 'openfoodfacts_ingredients.json'.")

except requests.exceptions.HTTPError as http_err:
    print(f"❌ Erreur HTTP : {http_err}")
except requests.exceptions.Timeout:
    print("❌ L'API OpenFoodFacts est trop lente ou indisponible. Réessaye plus tard.")
except requests.exceptions.RequestException as e:
    print(f"❌ Erreur réseau : {e}")
