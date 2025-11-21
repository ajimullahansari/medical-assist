from flask import Flask, request, render_template, jsonify, redirect, url_for, flash
import numpy as np
import pandas as pd
import pickle
import os
import math

# --- NEW IMPORTS FOR AUTHENTICATION ---
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash

# -------------------------------------

# ============================================================
# Flask App Setup & CONFIGURATION
# ============================================================
main = Flask(__name__)
main.config['SECRET_KEY'] = 'YourSuperSecretKeyForCollegeProject_12345'
main.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
main.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(main)
login_manager = LoginManager()
login_manager.init_app(main)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'


# ============================================================
# User Model for Database (NEW)
# ============================================================

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f"User('{self.username}', '{self.email}')"


# ============================================================
# Load Data & Model
# ============================================================

# Define base path for datasets and models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_PATH = os.path.join(BASE_DIR, "datasets")
MODELS_PATH = os.path.join(BASE_DIR, "models")


# Load datasets safely
def load_data(filename, default_value=pd.DataFrame()):
    path = os.path.join(DATASETS_PATH, filename)
    try:
        # यहाँ NaN वैल्यू को स्ट्रिंग के रूप में लोड होने से रोकने के लिए keep_default_na=True आवश्यक है
        return pd.read_csv(path, keep_default_na=True)
    except FileNotFoundError:
        print(f"Error: Dataset file '{path}' not found. Using empty DataFrame.")
        return default_value


sym_des = load_data("symtoms_df.csv")
precautions = load_data("precautions_df.csv")
workout = load_data("workout_df.csv")
description = load_data("description.csv")
medications = load_data('medications.csv')
diets = load_data("diets.csv")

# Load model safely
svc = None
model_path = os.path.join(MODELS_PATH, 'svc.pkl')
try:
    with open(model_path, 'rb') as f:
        svc = pickle.load(f)
except FileNotFoundError:
    print(f"Error: Model file '{model_path}' not found. Prediction will not work.")
except Exception as e:
    print(f"Error loading model: {e}")


# ============================================================
# Helper Functions and Dictionaries
# ============================================================

def helper(dis):
    # Description
    desc_series = description[description['Disease'] == dis]['Description']
    # FIX for ValueError: .iloc[0] का उपयोग करें
    dis_des = desc_series.iloc[0] if not desc_series.empty else "No description available."

    # Precautions
    pre_data = precautions[precautions['Disease'] == dis][
        ['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    # FIX: .values.tolist() का उपयोग करें
    pre = pre_data.values.tolist() if not pre_data.empty and len(pre_data) > 0 else [[]]

    # Medications
    med = medications[medications['Disease'] == dis]['Medication']
    medications_list = med.tolist() if not med.empty else []

    # Diets
    die = diets[diets['Disease'] == dis]['Diet']
    diets_list = die.tolist() if not die.empty else []

    # Workout
    wrkout = workout[workout['disease'] == dis]['workout']
    workout_list = wrkout.tolist() if not wrkout.empty else []

    return dis_des, pre, medications_list, diets_list, workout_list


symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4,
                 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9,
                 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13,
                 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18,
                 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22,
                 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27,
                 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32,
                 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37,
                 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42,
                 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46,
                 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50,
                 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54,
                 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58,
                 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61,
                 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66,
                 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70,
                 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74,
                 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78,
                 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82,
                 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86,
                 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89,
                 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92,
                 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96,
                 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100,
                 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103,
                 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107,
                 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110,
                 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113,
                 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116,
                 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120,
                 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124,
                 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127,
                 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction',
                 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma',
                 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)',
                 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A',
                 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis',
                 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)',
                 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism',
                 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis',
                 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection',
                 35: 'Psoriasis', 27: 'Impetigo'}


# Model Prediction function
def get_predicted_value(patient_symptoms):
    if svc is None:
        return "Model Not Loaded"

    if not patient_symptoms:
        return "No Symptoms Selected"

    input_vector = np.zeros(len(symptoms_dict))

    # Populate vector with 1s for selected symptoms
    for item in patient_symptoms:
        item = item.strip()
        if item and item in symptoms_dict:
            input_vector[symptoms_dict[item]] = 1
        elif item:
            print(f"Warning: Symptom '{item}' not found in symptoms_dict.")

    if np.sum(input_vector) == 0:
        return "Not enough valid symptoms selected for prediction."

    return diseases_list[svc.predict([input_vector])[0]]


# ============================================================
# Routes
# ============================================================

# --- NEW AUTHENTICATION ROUTES ---
@main.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('index')) # 'home' को 'index' से बदला गया

    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')

        user_exists = User.query.filter_by(email=email).first()
        if user_exists:
            flash('यह ईमेल पहले से पंजीकृत है। कृपया लॉगिन करें या दूसरा ईमेल उपयोग करें।', 'danger')
            return redirect(url_for('signup'))

        new_user = User(username=username, email=email)
        new_user.set_password(password)

        db.session.add(new_user)
        db.session.commit()

        flash('पंजीकरण सफल रहा! अब आप लॉगिन कर सकते हैं।', 'success')
        return redirect(url_for('login'))

    return render_template('signup.html')


@main.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index')) # 'home' को 'index' से बदला गया

    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        user = User.query.filter_by(email=email).first()

        if user and user.check_password(password):
            login_user(user, remember=True)
            flash(f'आप सफलतापूर्वक लॉगिन हो गए हैं! {user.username} जी, आपका स्वागत है।', 'success')
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('index')) # 'home' को 'index' से बदला गया
        else:
            flash('लॉगिन असफल। कृपया ईमेल या पासवर्ड जांचें।', 'danger')

    return render_template('login.html')


@main.route('/logout')
@login_required
def logout():
    logout_user()
    flash('आप सफलतापूर्वक लॉगआउट हो गए हैं।', 'info')
    return redirect(url_for('index')) # 'home' को 'index' से बदला गया


# --- END AUTHENTICATION ROUTES ---


@main.route("/", methods=['GET']) # सुनिश्चित करें कि यह केवल GET है
@login_required
def index():
    return render_template("index.html")


@main.route('/predict', methods=['POST']) # सुनिश्चित करें कि यह केवल POST है
@login_required
def home():
    # यह फंक्शन अब सिर्फ POST रिक्वेस्ट संभालेगा
    symptoms = request.form.get('symptoms', '').strip()

    # 1. Empty input check
    if not symptoms:
        message = "कृपया बीमारी का अनुमान लगाने के लिए कम से कम एक लक्षण चुनें।"
        return render_template('index.html', message=message)

    # 2. Convert input to list
    user_symptoms = [s.strip() for s in symptoms.split(',')]

    # 3. Clean up the list
    user_symptoms = [symptom for symptom in user_symptoms if symptom]

    # 4. Prediction
    predicted_disease = get_predicted_value(user_symptoms)

    # 5. Check Prediction status
    if predicted_disease in ["No Symptoms Selected", "Not enough valid symptoms selected for prediction.",
                             "Model Not Loaded"]:
        message = predicted_disease + " कृपया अधिक वैध लक्षण चुनें।"
        if predicted_disease == "Model Not Loaded":
            message = "AI मॉडल लोड नहीं हो सका। कृपया अपनी 'models' फ़ोल्डर की जाँच करें।"
        return render_template('index.html', message=message)

    # 6. Get additional information
    dis_des, precautions_data, medications_list, rec_diet, workout = helper(predicted_disease)

    my_precautions = []
    # FIX for Attribute Error: Checking if item is a string before calling .strip()
    # precautions_data is a list of lists: [['pre1', 'pre2', 'pre3', 'pre4']]
    if len(precautions_data) > 0 and len(precautions_data[0]) > 0:
        for i in precautions_data[0]:
            # FIX: Check for string type before using .strip() to avoid 'float' object error
            # यह FIX आपके Attribute Error को हल करता है
            if isinstance(i, str) and i.strip():
                my_precautions.append(i.strip())

    return render_template('index.html', predicted_disease=predicted_disease, dis_des=dis_des,
                           my_precautions=my_precautions, medications=medications_list, my_diet=rec_diet,
                           workout=workout)


# about view funtion and path
@main.route('/about')
def about():
    return render_template("about.html")


# contact view funtion and path
@main.route('/contact')
def contact():
    return render_template("contact.html")


# developer view funtion and path
@main.route('/developer')
def developer():
    return render_template("developer.html")


# about view funtion and path
@main.route('/blog')
def blog():
    return render_template("blog.html")


if __name__ == '__main__':
    # डेटाबेस बनाने के लिए यह कोड सुनिश्चित करें
    with main.app_context():
        db.create_all()

    # main.run() को ब्लॉक के बाहर होना चाहिए!
    main.run(debug=True)




# old wala=============================== start here ...............................
# from flask import Flask, request, render_template, jsonify
# import numpy as np
# import pandas as pd
# import pickle
# import os
# import math  # math module imported to check for NaN (just in case)
#
# # flask app
# app = Flask(__name__)
#
# # ============================================================
# # Load Data & Model
# # ============================================================
#
# # Define base path for datasets and models
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# DATASETS_PATH = os.path.join(BASE_DIR, "datasets")
# MODELS_PATH = os.path.join(BASE_DIR, "models")
#
#
# # Load datasets safely
# def load_data(filename, default_value=pd.DataFrame()):
#     path = os.path.join(DATASETS_PATH, filename)
#     # NaN values को स्ट्रिंग 'nan' के रूप में लोड न करें, उन्हें None या float NaN के रूप में रहने दें
#     try:
#         return pd.read_csv(path)
#     except FileNotFoundError:
#         print(f"Error: Dataset file '{path}' not found. Using empty DataFrame.")
#         return default_value
#
#
# sym_des = load_data("symtoms_df.csv")
# precautions = load_data("precautions_df.csv")
# workout = load_data("workout_df.csv")
# description = load_data("description.csv")
# medications = load_data('medications.csv')
# diets = load_data("diets.csv")
#
# # Load model safely
# svc = None
# model_path = os.path.join(MODELS_PATH, 'svc.pkl')
# try:
#     with open(model_path, 'rb') as f:
#         svc = pickle.load(f)
# except FileNotFoundError:
#     print(f"Error: Model file '{model_path}' not found. Prediction will not work.")
# except Exception as e:
#     print(f"Error loading model: {e}")
#
#
# # ============================================================
# # Helper Functions and Dictionaries
# # ============================================================
#
# def helper(dis):
#     # Description
#     desc = description[description['Disease'] == dis]['Description']
#     dis_des = desc.iloc[0] if not desc.empty else "No description available."
#
#     # Precautions
#     pre_data = precautions[precautions['Disease'] == dis][
#         ['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
#     pre = pre_data.values.tolist() if not pre_data.empty else [[]]
#
#     # Medications
#     med = medications[medications['Disease'] == dis]['Medication']
#     medications_list = med.tolist() if not med.empty else []
#
#     # Diets
#     die = diets[diets['Disease'] == dis]['Diet']
#     diets_list = die.tolist() if not die.empty else []
#
#     # Workout
#     wrkout = workout[workout['disease'] == dis]['workout']
#     workout_list = wrkout.tolist() if not wrkout.empty else []
#
#     return dis_des, pre, medications_list, diets_list, workout_list
#
#
# symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4,
#                  'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9,
#                  'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13,
#                  'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18,
#                  'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22,
#                  'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27,
#                  'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32,
#                  'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37,
#                  'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42,
#                  'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46,
#                  'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50,
#                  'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54,
#                  'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58,
#                  'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61,
#                  'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66,
#                  'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70,
#                  'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74,
#                  'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78,
#                  'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82,
#                  'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86,
#                  'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89,
#                  'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92,
#                  'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96,
#                  'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100,
#                  'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103,
#                  'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107,
#                  'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110,
#                  'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113,
#                  'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116,
#                  'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120,
#                  'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124,
#                  'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127,
#                  'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
# diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction',
#                  33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma',
#                  23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)',
#                  28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A',
#                  19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis',
#                  36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)',
#                  18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism',
#                  25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis',
#                  0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection',
#                  35: 'Psoriasis', 27: 'Impetigo'}
#
#
# # Model Prediction function
# def get_predicted_value(patient_symptoms):
#     if svc is None:
#         return "Model Not Loaded"
#
#     if not patient_symptoms:
#         return "No Symptoms Selected"
#
#     input_vector = np.zeros(len(symptoms_dict))
#
#     # Populate vector with 1s for selected symptoms
#     for item in patient_symptoms:
#         item = item.strip()
#         if item and item in symptoms_dict:
#             input_vector[symptoms_dict[item]] = 1
#         elif item:
#             print(f"Warning: Symptom '{item}' not found in symptoms_dict.")
#
#     if np.sum(input_vector) == 0:
#         return "Not enough valid symptoms selected for prediction."
#
#     return diseases_list[svc.predict([input_vector])[0]]
#
#
# # ============================================================
# # Routes
# # ============================================================
#
# @app.route("/")
# def index():
#     return render_template("index.html")
#
#
# @app.route('/predict', methods=['GET', 'POST'])
# def home():
#     if request.method == 'POST':
#         symptoms = request.form.get('symptoms', '').strip()
#
#         # 1. Empty input check
#         if not symptoms:
#             message = "कृपया बीमारी का अनुमान लगाने के लिए कम से कम एक लक्षण चुनें।"
#             return render_template('index.html', message=message)
#
#         # 2. Convert input to list
#         user_symptoms = [s.strip() for s in symptoms.split(',')]
#
#         # 3. Clean up the list
#         user_symptoms = [symptom for symptom in user_symptoms if symptom]
#
#         # 4. Prediction
#         predicted_disease = get_predicted_value(user_symptoms)
#
#         # 5. Check Prediction status
#         if predicted_disease in ["No Symptoms Selected", "Not enough valid symptoms selected for prediction.",
#                                  "Model Not Loaded"]:
#             message = predicted_disease + " कृपया अधिक वैध लक्षण चुनें।"
#             if predicted_disease == "Model Not Loaded":
#                 message = "AI मॉडल लोड नहीं हो सका। कृपया अपनी 'models' फ़ोल्डर की जाँच करें।"
#             return render_template('index.html', message=message)
#
#         # 6. Get additional information
#         dis_des, precautions_data, medications_list, rec_diet, workout = helper(predicted_disease)
#
#         my_precautions = []
#         # FIX for Attribute Error: Checking if item is a string before calling .strip()
#         # precautions_data is a list of lists: [['pre1', 'pre2', 'pre3', 'pre4']]
#         if len(precautions_data) > 0 and len(precautions_data[0]) > 0:
#             for i in precautions_data[0]:
#                 # FIX 2: Check for string type before using .strip() to avoid 'float' object error
#                 if isinstance(i, str) and i.strip():
#                     my_precautions.append(i.strip())
#
#         return render_template('index.html', predicted_disease=predicted_disease, dis_des=dis_des,
#                                my_precautions=my_precautions, medications=medications_list, my_diet=rec_diet,
#                                workout=workout)
#
#     return render_template('index.html')
#
#
# # about view funtion and path
# @app.route('/about')
# def about():
#     return render_template("about.html")
#
#
# # contact view funtion and path
# @app.route('/contact')
# def contact():
#     return render_template("contact.html")
#
#
# # developer view funtion and path
# @app.route('/developer')
# def developer():
#     return render_template("developer.html")
#
#
# # about view funtion and path
# @app.route('/blog')
# def blog():
#     return render_template("blog.html")
#
#
# if __name__ == '__main__':
#     app.run(debug=True)