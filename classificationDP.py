import streamlit as st
from tensorflow import keras
from PIL import Image
import numpy as np
from functions import *
from tensorflow.keras.applications.resnet50 import preprocess_input

# Charger le modèle Keras à partir du fichier .h5
model = keras.models.load_model('ResNet50_DEL2.h5')

# Interface utilisateur Streamlit
st.title('Classification Deep learning')

# Ajout de la fonctionnalité d'upload d'image
uploaded_file = st.file_uploader("Téléchargez une image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Afficher l'image téléchargée
    image = Image.open(uploaded_file)
    st.image(image, caption="Image téléchargée", use_column_width=True)

    # Prétraitement de l'image pour l'adapter au modèle
    image = image.resize((224, 224))
    image_array = np.array(image)
    image_array = preprocess_input(image_array)
    image_array = np.expand_dims(image_array, axis=0)

    # Faire des prédictions avec le modèle
    predictions = model.predict(image_array)

    # Afficher les résultats
    st.write('Prédictions :')
    labels = ['Brown bear', 'Butterfly', 'Canary',
       'Caterpillar', 'Centipede', 'Cheetah', 'Chicken', 'Crab',
       'Crocodile', 'Deer', 'Duck', 'Eagle', 'Elephant', 'Fish', 'Fox',
       'Frog', 'Giraffe', 'Goat', 'Goldfish', 'Goose',
       'Harbor seal', 'Horse', 'Jaguar',
       'Jellyfish', 'Ladybug', 'Leopard', 'Lion',
       'Lizard', 'Monkey', 'Moths and butterflies',
       'Mouse', 'Ostrich', 'Owl', 'Parrot',
       'Penguin', 'Pig', 'Polar bear', 'Rabbit', 'Raccoon',
       'Rhinoceros', 'Sea lion', 'Sea turtle',
       'Shark', 'Snail', 'Snake',
       'Sparrow', 'Spider', 'Squirrel', 'Starfish', 'Swan',
       'Tiger', 'Tortoise', 'Whale',
       'Woodpecker', 'Worm', 'Zebra']
    
    for i, proba in enumerate(predictions[0]):
        # Trouver l'index de la classe avec le pourcentage le plus élevé
        max_index = np.argmax(predictions[0])
        max_class = labels[max_index]
        max_proba = predictions[0][max_index] * 100

    # Afficher la classe avec le pourcentage le plus élevé
    st.write(f'Classe prédite : {max_class}')
    st.write(f'Probabilité : {max_proba:.2f}%')

