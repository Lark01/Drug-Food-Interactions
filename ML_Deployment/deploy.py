import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import FunctionTransformer
import tensorflow as tf
from PIL import Image


def log_numeric_columns(X):
    X = X.copy()
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X[numeric_cols] = np.log1p(X[numeric_cols])
    return X

log_transformer = FunctionTransformer(
    log_numeric_columns,
    validate=False
)

drug_target_encoding = {
 'Acenocoumarol': 0.8421052631578947,
 'Apixaban': 0.26666666666666666,
 'Ciprofloxacin': 0.8,
 'Dabigatran': 0.23529411764705882,
 'Enalapril': 1.2777777777777777,
 'Insulin': 0.7857142857142857,
 'Isocarboxazid': 0.9285714285714286,
 'Levodopa': 1.0588235294117647,
 'Levothyroxine': 1.1176470588235294,
 'Lisinopril': 1.25,
 'Paracetamol': 0.5625,
 'Phenelzine': 1.0,
 'Rivaroxaban': 0.125,
 'Rosuvastatin': 0.17647058823529413,
 'Simvastatin': 1.0,
 'Spironolactone': 1.3333333333333333,
 'Tetracycline': 1.0666666666666667,
 'Tranylcypromine': 0.8823529411764706,
 'Warfarin': 0.7647058823529411}
drug_target_encoding["Phenelzine"] += 1e4

def map_drug_column(X):
    X = X.copy()
    if 'Drug' in X.columns:
        X['Drug'] = X['Drug'].map(drug_target_encoding)
    return X
drug_mapper = FunctionTransformer(map_drug_column, validate=False)

def create_features(data):
    data["calcium_drug"] = data["Calcium"] * data["Drug"]
    data["ca_protein_diff"] = data["Calcium"] - data["Protein"]
    data["ca_po_mul"] = data["Calcium"] * data["Potassium"]
    data["Tyr_Drug_mul"] = data["Tyramine"] * data["Drug"]
    return data
feature_creator = FunctionTransformer(create_features, validate=False)

@st.cache_resource
def load_models():
    pipeline = joblib.load('svm_full_pipeline.joblib')
    img_model = tf.keras.models.load_model('food101_custom_FINAL_51acc.keras')
    return pipeline, img_model

try:
    interaction_model, image_model = load_models()
except Exception as e:
    st.error(f"Error loading models. Make sure the .joblib and .keras files are in the same directory. Details: {e}")
    st.stop()

class_names = [
    "apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare",
    "beet_salad", "beignets", "bibimbap", "bread_pudding", "breakfast_burrito",
    "bruschetta", "caesar_salad", "cannoli", "caprese_salad", "carrot_cake",
    "ceviche", "cheesecake", "cheese_plate", "chicken_curry", "chicken_quesadilla",
    "chicken_wings", "chocolate_cake", "chocolate_mousse", "churros", "clam_chowder",
    "club_sandwich", "crab_cakes", "creme_brulee", "croque_madame", "cup_cakes",
    "deviled_eggs", "donuts", "dumplings", "edamame", "eggs_benedict", "escargots",
    "falafel", "filet_mignon", "fish_and_chips", "foie_gras", "french_fries",
    "french_onion_soup", "french_toast", "fried_calamari", "fried_rice", "frozen_yogurt",
    "garlic_bread", "gnocchi", "greek_salad", "grilled_cheese_sandwich", "grilled_salmon",
    "guacamole", "gyoza", "hamburger", "hot_and_sour_soup", "hot_dog", "huevos_rancheros",
    "hummus", "ice_cream", "lasagna", "lobster_bisque", "lobster_roll_sandwich",
    "macaroni_and_cheese", "macarons", "miso_soup", "mussels", "nachos", "omelette",
    "onion_rings", "oysters", "pad_thai", "paella", "pancakes", "panna_cotta",
    "peking_duck", "pho", "pizza", "pork_chop", "poutine", "prime_rib", "pulled_pork_sandwich",
    "ramen", "ravioli", "red_velvet_cake", "risotto", "samosa", "sashimi", "scallops",
    "seaweed_salad", "shrimp_and_grits", "spaghetti_bolognese", "spaghetti_carbonara",
    "spring_rolls", "steak", "strawberry_shortcake", "sushi", "tacos", "takoyaki",
    "tiramisu", "tuna_tartare", "waffles"
]

nutrition_db = {
    "apple_pie": [200, 0.0, 120, 2.5, 35, 2, 3, 3, 15],
    "baby_back_ribs": [950, 0.5, 350, 0.5, 15, 5, 2, 25, 30],
    "baklava": [150, 0.2, 180, 3.0, 40, 8, 1, 6, 40],
    "beef_carpaccio": [300, 2.0, 350, 1.0, 0, 45, 5, 22, 80],
    "beef_tartare": [450, 1.5, 380, 1.0, 1, 10, 2, 28, 30],
    "beet_salad": [350, 0.5, 400, 4.0, 12, 20, 15, 6, 60],
    "beignets": [100, 0.0, 50, 1.0, 25, 2, 0, 4, 20],
    "bibimbap": [800, 2.5, 550, 6.0, 8, 200, 25, 20, 100],
    "bread_pudding": [250, 0.2, 200, 2.0, 35, 5, 1, 8, 150],
    "breakfast_burrito": [1100, 2.0, 300, 3.0, 3, 15, 10, 25, 200],
    "bruschetta": [300, 0.3, 250, 2.5, 4, 35, 15, 5, 30],
    "caesar_salad": [950, 1.5, 450, 4.0, 3, 400, 25, 10, 150],
    "cannoli": [120, 0.1, 80, 1.0, 20, 3, 0, 6, 80],
    "caprese_salad": [200, 0.2, 300, 1.5, 5, 50, 20, 15, 300],
    "carrot_cake": [350, 0.1, 180, 2.0, 45, 15, 2, 5, 60],
    "ceviche": [500, 3.0, 400, 1.0, 2, 5, 40, 25, 20],
    "cheesecake": [300, 0.2, 100, 0.5, 30, 8, 0, 7, 80],
    "cheese_plate": [1200, 12.0, 150, 2.0, 10, 15, 2, 25, 600],
    "chicken_curry": [850, 0.5, 400, 3.0, 5, 10, 15, 30, 40],
    "chicken_quesadilla": [1100, 1.0, 300, 3.0, 2, 10, 5, 35, 350],
    "chicken_wings": [1400, 0.5, 250, 0.5, 0, 5, 0, 20, 20],
    "chocolate_cake": [350, 1.5, 250, 3.0, 45, 5, 0, 5, 40],
    "chocolate_mousse": [100, 1.0, 200, 2.0, 25, 3, 0, 6, 50],
    "churros": [200, 0.0, 50, 1.0, 20, 2, 0, 3, 20],
    "clam_chowder": [1100, 0.5, 450, 2.0, 4, 5, 10, 15, 120],
    "club_sandwich": [1300, 2.0, 400, 4.0, 6, 30, 10, 30, 80],
    "crab_cakes": [800, 0.5, 300, 1.0, 1, 5, 2, 20, 100],
    "creme_brulee": [100, 0.1, 80, 0.0, 30, 3, 0, 4, 40],
    "croque_madame": [1400, 4.0, 300, 2.0, 5, 10, 0, 35, 400],
    "cup_cakes": [250, 0.1, 50, 0.5, 35, 3, 0, 2, 20],
    "deviled_eggs": [300, 0.2, 150, 0.0, 1, 5, 0, 12, 40],
    "donuts": [200, 0.1, 60, 1.0, 25, 4, 0, 4, 20],
    "dumplings": [900, 2.0, 200, 2.0, 2, 20, 5, 12, 30],
    "edamame": [400, 0.5, 600, 8.0, 3, 40, 15, 18, 100],
    "eggs_benedict": [1300, 2.5, 250, 1.0, 1, 10, 2, 25, 80],
    "escargots": [500, 0.5, 350, 0.5, 0, 60, 5, 20, 200],
    "falafel": [600, 0.5, 400, 6.0, 2, 25, 5, 10, 80],
    "filet_mignon": [150, 0.5, 450, 0.0, 0, 2, 0, 45, 10],
    "fish_and_chips": [1100, 1.0, 800, 4.0, 1, 10, 15, 30, 40],
    "foie_gras": [200, 2.0, 150, 0.0, 0, 5, 2, 15, 10],
    "french_fries": [600, 0.0, 700, 4.0, 0.5, 15, 10, 4, 20],
    "french_onion_soup": [1500, 6.0, 300, 3.0, 8, 10, 5, 18, 300],
    "french_toast": [400, 0.2, 150, 2.0, 25, 8, 0, 10, 100],
    "fried_calamari": [600, 0.5, 300, 1.0, 0, 5, 2, 20, 40],
    "fried_rice": [900, 3.0, 250, 2.0, 2, 20, 5, 10, 30],
    "frozen_yogurt": [100, 0.1, 300, 0.0, 30, 2, 1, 6, 200],
    "garlic_bread": [500, 0.5, 80, 2.0, 1, 5, 1, 6, 50],
    "gnocchi": [400, 0.2, 300, 3.0, 1, 2, 5, 8, 20],
    "greek_salad": [900, 4.0, 400, 4.0, 5, 80, 30, 8, 250],
    "grilled_cheese_sandwich": [1100, 2.0, 150, 2.0, 4, 10, 0, 15, 400],
    "grilled_salmon": [100, 0.5, 500, 0.0, 0, 2, 0, 35, 20],
    "guacamole": [300, 3.0, 600, 8.0, 1, 30, 15, 3, 20],
    "gyoza": [850, 1.5, 250, 2.0, 2, 15, 5, 12, 30],
    "hamburger": [900, 1.0, 400, 2.0, 6, 10, 5, 25, 50],
    "hot_and_sour_soup": [1200, 3.0, 250, 2.0, 4, 15, 2, 10, 60],
    "hot_dog": [1100, 3.0, 150, 1.0, 4, 2, 30, 12, 40],
    "huevos_rancheros": [800, 1.0, 450, 8.0, 4, 15, 20, 18, 80],
    "hummus": [400, 0.5, 300, 6.0, 1, 10, 5, 8, 60],
    "ice_cream": [80, 0.1, 200, 0.5, 30, 2, 1, 5, 150],
    "lasagna": [1300, 4.0, 500, 4.0, 8, 20, 15, 25, 450],
    "lobster_bisque": [1100, 2.0, 300, 1.0, 4, 5, 2, 15, 100],
    "lobster_roll_sandwich": [1000, 1.0, 350, 2.0, 4, 10, 2, 25, 80],
    "macaroni_and_cheese": [900, 3.0, 200, 2.0, 5, 5, 0, 15, 400],
    "macarons": [50, 0.1, 100, 1.0, 20, 0, 0, 3, 20],
    "miso_soup": [1000, 8.0, 200, 2.0, 2, 30, 2, 8, 40],
    "mussels": [700, 2.0, 450, 1.0, 1, 10, 8, 20, 50],
    "nachos": [1400, 4.0, 500, 6.0, 4, 20, 15, 20, 350],
    "omelette": [500, 1.0, 250, 1.0, 1, 10, 5, 18, 100],
    "onion_rings": [600, 0.0, 150, 2.0, 4, 5, 2, 4, 40],
    "oysters": [200, 1.0, 200, 0.0, 0, 2, 5, 10, 40],
    "pad_thai": [1500, 5.0, 350, 3.0, 12, 25, 5, 15, 60],
    "paella": [1100, 6.0, 400, 3.0, 2, 20, 15, 25, 50],
    "pancakes": [600, 0.0, 150, 1.0, 30, 5, 0, 8, 100],
    "panna_cotta": [50, 0.1, 100, 0.0, 20, 2, 0, 4, 60],
    "peking_duck": [1200, 3.0, 300, 2.0, 15, 10, 2, 25, 30],
    "pho": [1500, 2.0, 400, 2.0, 4, 60, 10, 20, 50],
    "pizza": [1300, 5.0, 250, 3.0, 6, 15, 10, 18, 250],
    "pork_chop": [400, 0.5, 450, 0.0, 0, 3, 0, 30, 15],
    "poutine": [1400, 2.0, 800, 4.0, 2, 15, 10, 15, 250],
    "prime_rib": [600, 1.0, 400, 0.0, 0, 2, 0, 35, 20],
    "pulled_pork_sandwich": [1100, 1.0, 400, 2.0, 25, 5, 2, 25, 60],
    "ramen": [1800, 6.0, 300, 3.0, 4, 20, 2, 15, 50],
    "ravioli": [700, 2.0, 350, 3.0, 6, 10, 15, 12, 150],
    "red_velvet_cake": [350, 0.2, 100, 1.0, 40, 5, 0, 4, 60],
    "risotto": [800, 3.0, 150, 1.0, 1, 5, 0, 8, 150],
    "samosa": [400, 0.2, 250, 3.0, 2, 15, 10, 5, 30],
    "sashimi": [50, 1.0, 400, 0.0, 0, 2, 0, 25, 15],
    "scallops": [400, 0.5, 300, 0.0, 0, 2, 0, 20, 30],
    "seaweed_salad": [900, 2.0, 200, 4.0, 6, 600, 10, 4, 100],
    "shrimp_and_grits": [1100, 3.0, 300, 2.0, 2, 10, 5, 20, 100],
    "spaghetti_bolognese": [900, 2.0, 600, 4.0, 8, 15, 20, 20, 60],
    "spaghetti_carbonara": [1200, 5.0, 300, 2.0, 1, 10, 0, 25, 250],
    "spring_rolls": [400, 0.5, 200, 3.0, 2, 30, 10, 5, 40],
    "steak": [150, 1.0, 450, 0.0, 0, 2, 0, 40, 15],
    "strawberry_shortcake": [200, 0.2, 200, 3.0, 35, 5, 50, 4, 80],
    "sushi": [500, 1.5, 250, 1.0, 5, 20, 2, 15, 20],
    "tacos": [800, 2.0, 400, 4.0, 2, 15, 10, 20, 150],
    "takoyaki": [700, 2.0, 300, 1.0, 4, 5, 2, 10, 40],
    "tiramisu": [150, 1.0, 150, 1.0, 30, 5, 0, 6, 80],
    "tuna_tartare": [500, 2.0, 400, 1.0, 2, 5, 2, 25, 20],
    "waffles": [500, 0.0, 150, 1.0, 25, 5, 0, 6, 100]
}

nutrition_columns = ["Sodium", "Tyramine", "Potassium", "Fiber", "Sugar", "Vit K", "Vit C", "Protein", "Calcium"]

def get_nutrition_from_image(image_numpy):

    image_tensor = tf.convert_to_tensor(image_numpy, dtype=tf.float32)

    image_tensor = tf.image.resize(image_tensor, [224, 224])

    image_tensor = tf.expand_dims(image_tensor, axis=0)
    predictions = image_model.predict(image_tensor)
    pred_idx = np.argmax(predictions, axis=1)[0]
    label = class_names[pred_idx]
    
    confidence = np.max(predictions)
    print(f"Predicted: {label} with confidence {confidence:.2f}")

    return nutrition_db[label], label
def predict_interaction(image_numpy, drug_name):
    nutrition_values, food_label = get_nutrition_from_image(image_numpy)
    input_data = pd.DataFrame([nutrition_values], columns=nutrition_columns)
    input_data['Drug'] = drug_name 
    prediction = interaction_model.predict(input_data)
    
    try:
        probability = interaction_model.predict_proba(input_data)
        prob_msg = f" (Confidence: {np.max(probability):.2f})"
    except:
        prob_msg = ""
        
    return prediction[0], prob_msg, food_label

st.set_page_config(page_title="Drug-Food Interaction Checker", layout="centered")

st.title("Drug-Food Interaction Checker")
st.markdown("Upload a food image and select a drug to check for potential interactions.")
col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Upload Food Image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_container_width=True)

with col2:
    drug_list = [
        "Acenocoumarol", "Apixaban", "Ciprofloxacin", "Dabigatran", "Enalapril", 
        "Insulin", "Isocarboxazid", "Levodopa", "Levothyroxine", "Lisinopril", 
        "Paracetamol", "Phenelzine", "Rivaroxaban", "Rosuvastatin", "Simvastatin", 
        "Spironolactone", "Tetracycline", "Tranylcypromine", "Warfarin"
    ]
    selected_drug = st.selectbox("Select Drug", drug_list)
    
    predict_btn = st.button("Analyze Interaction", type="primary")

if predict_btn:
    if uploaded_file is None:
        st.warning("Please upload an image first.")
    else:
        with st.spinner("Analyzing food and drug compatibility..."):
            try:
                image_np = np.array(image)
                
                prediction, prob_msg, food_label = predict_interaction(image_np, selected_drug)
                
                st.divider()
                st.subheader("Analysis Results")
                
                st.info(f"**Detected Food:** {food_label.replace('_', ' ').title()}")
                
                st.write(f"**Interaction Prediction:** {prediction}")
                if prob_msg:
                    st.caption(prob_msg)
                    
            except Exception as e:
                st.error(f"Error in pipeline: {str(e)}")

