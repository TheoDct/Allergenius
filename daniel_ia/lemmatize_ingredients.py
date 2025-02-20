import pandas as pd
import spacy
from tqdm import tqdm

# Charger le modèle SpaCy avec un thread unique pour éviter les conflits mémoire
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
nlp.max_length = 1500000  # Augmente la capacité de traitement de texte

# Lire les ingrédients et forcer une copie pour éviter les problèmes de mémoire
df = pd.read_csv("huggingface_recipes.csv", usecols=["ingredients"]).copy()

# Lemmatisation avec `nlp.pipe()` pour optimiser le traitement batch
def lemmatize_ingredients(ingredients_list):
    lemmatized_list = []
    for doc in tqdm(nlp.pipe(ingredients_list, batch_size=500, n_process=1), total=len(ingredients_list), desc="🔄 Lemmatisation"):
        lemmatized_list.append(" ".join([token.lemma_ for token in doc]))
    return lemmatized_list

# Appliquer la lemmatisation
df["ingredients"] = lemmatize_ingredients(df["ingredients"].astype(str))

# Sauvegarde du fichier lemmatisé
df.to_csv("huggingface_recipes_lemmatized.csv", index=False)
print("✅ Fichier lemmatisé sauvegardé sous 'huggingface_recipes_lemmatized.csv'")
