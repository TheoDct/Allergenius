import pickle 
import numpy as np

def predict_recipe(recipe_text):
    model = pickle.load(open('./model/model.pkl', 'rb'))
    tfidf = pickle.load(open('./model/tfidf.pkl', 'rb'))
    mlb = pickle.load(open('./model/mlb.pkl', 'rb'))
    
    recipe_vec = tfidf.transform([recipe_text])
    prediction = model.predict(recipe_vec)
    predicted_allergies = mlb.inverse_transform(prediction)[0]

    return {"Predicted_Allergens": predicted_allergies}

# Exemple d'utilisation
print(predict_recipe("1 cup of flour, 2 eggs, 3 cups of sugar, 1 teaspoon of vanilla extract"))  
print(predict_recipe("200g of chicken breast, 1 tablespoon of olive oil, 1 teaspoon of salt, 1/2 teaspoon of black pepper"))
print(predict_recipe("1 cup of rice, 2 cups of water, 1 teaspoon of salt, 1 tablespoon of butter"))
print(predict_recipe("100g of pasta, 50g of cheese, 1/2 cup of milk, 1 teaspoon of garlic powder"))
print(predict_recipe("2 slices of bread, 1 tablespoon of peanut butter, 1 tablespoon of jelly"))
