import pickle
from flask import Flask, request, render_template, url_for
import numpy as np
import pandas as pd
from src.logger import logging
import webbrowser
import threading

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline1 import CustomData, PredictPipeline
from src.pipeline.predict_pipeline2 import CustomData2, PredictPipeline2

application = Flask(__name__)
app = application

# Route for the Home Page
@app.route('/')
def index():
    return render_template('index.html')

def open_browser():
    webbrowser.open_new('http://localhost:5000/')

@app.route('/about_parkinsons')
def about_gdm():
    return render_template('about_parkinsons.html')

@app.route('/faq')
def faq():
    return render_template('faq.html')

@app.route('/contact_us')
def contact_us():
    return render_template('contact_us.html')

@app.route('/predictdata1', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('prediction.html', results=None)
    else:
        data = CustomData(
            Dribbling_of_saliva_during_the_daytime=int(request.form.get('Dribbling_of_saliva_during_the_daytime')),
            height=int(request.form.get('height')),
            Falling=int(request.form.get('Falling')),
            Memory_problems=int(request.form.get('Problems_remembering_things_that_have_happened_recently_or_forgetting_to_do_things')),
            Loss_of_taste_smell=int(request.form.get('Loss_or_change_in_your_ability_to_taste_or_smell')),
            Effect_of_alcohol=int(request.form.get('effect_of_alcohol_on_tremor')),
            Acting_out_dreams=int(request.form.get('Talking_or_moving_about_in_your_sleep_as_if_you_are_acting_out_a_dream')),
            Difficulty_swallowing=int(request.form.get('Difficulty_swallowing_food_or_drink_or_problems_with_choking')),
            Constipation=int(request.form.get('Constipation_less_than_3_bowel_movements_a_week_or_having_to_strain_to_pass_a_stool')),
            gender=int(request.form.get('gender')),
            weight=int(request.form.get('weight')),
            Night_urination=int(request.form.get('Getting_up_regularly_at_night_to_pass_urine')),
            Urgency_to_urinate=int(request.form.get('A_sense_of_urgency_to_pass_urine_makes_you_rush_to_the_toilet')),
            age_at_diagnosis=int(request.form.get('age_at_diagnosis'))
        )

        pred_df = data.get_data_as_data_frame()
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        print("Model Prediction Output:", results)  # Debugging line

        if results[0] == 0.0:
            msg = "You are a Healthy Person!"
        elif(results[0] > 0.0):
            msg = "You seem to be unhealthy. Let's verify if you have Parkinson's disease or not!"

        logging.info(f"Predicted result: {results[0]}, Message: {msg}")

        # **Pass the data dictionary to the next page**
        return render_template('post.html', results=msg, previous_features=data.__dict__)

@app.route('/predictdata2', methods=['POST'])
def predict_datapoint2():
    try:
        # Get previous 14 features
        previous_features = {
            'height': int(request.form.get('height', 0)),
            'Dribbling of saliva during the daytime': int(request.form.get('Dribbling_of_saliva_during_the_daytime', 0)),
            'Falling': int(request.form.get('Falling', 0)),
            'Problems remembering things that have happened recently or forgetting to do things': int(request.form.get('Memory_problems', 0)),
            'Loss or change in your ability to taste or smell': int(request.form.get('Loss_of_taste_smell', 0)),
            'effect_of_alcohol_on_tremor': int(request.form.get('Effect_of_alcohol', 0)),
            'Talking or moving about in your sleep as if you are acting out a dream': int(request.form.get('Acting_out_dreams', 0)),
            'Difficulty swallowing food or drink or problems with choking': int(request.form.get('Difficulty_swallowing', 0)),
            'Constipation (less than 3 bowel movements a week) or having to strain to pass a stool (faeces)': int(request.form.get('Constipation', 0)),
            'gender': int(request.form.get('gender', 0)),
            'weight': int(request.form.get('weight', 0)),
            'Getting up regularly at night to pass urine': int(request.form.get('Night_urination', 0)),
            'A sense of urgency to pass urine makes you rush to the toilet': int(request.form.get('Urgency_to_urinate', 0)),
            'age_at_diagnosis': int(request.form.get('age_at_diagnosis', 0))
        }

        # Get new 15 features
        new_features = {
            'Difficulty concentrating or staying focussed': int(request.form.get('Difficulty_concentrating_or_staying_focused', 0)),
            'Feeling sad low or blue': int(request.form.get('Feeling_sad_low_or_blue', 0)),
            'Difficulty getting to sleep at night or staying asleep at night': int(request.form.get('Difficulty_getting_to_sleep_at_night_or_staying_asleep_at_night', 0)),
            'Unpleasant sensations in your legs at night or while resting and a feeling that you need to move': int(request.form.get('Unpleasant_sensations_in_legs_at_night_or_while_resting_and_a_feeling_that_you_need_to_move', 0)),
            'Finding it difficult to have sex when you try': int(request.form.get('Finding_it_difficult_to_have_sex_when_you_try', 0)),
            'Finding it difficult to stay awake during activities such as working driving or eating': int(request.form.get('Finding_it_difficult_to_stay_awake_during_activities_such_as_working_driving_or_eating', 0)),
            'Feeling that your bowel emptying is incomplete after having been to the toilet': int(request.form.get('Feeling_that_your_bowel_emptying_is_incomplete_after_having_been_to_the_toilet', 0)),
            'Intense vivid dreams or frightening dreams': int(request.form.get('Intense_vivid_dreams_or_frightening_dreams', 0)),
            'Unexplained pains (not due to known conditions such as arthritis)': int(request.form.get('Unexplained_pains_not_due_to_known_conditions_such_as_arthritis', 0)),
            'Excessive sweating': int(request.form.get('Excessive_sweating', 0)),
            'Swelling of your legs': int(request.form.get('Swelling_of_your_legs', 0)),
            'Bowel (fecal) incontinence': int(request.form.get('Bowel_fecal_incontinence', 0)),
            'Feeling light headed dizzy or weak standing from sitting or lying': int(request.form.get('Feeling_light_headed_dizzy_or_weak_standing_from_sitting_or_lying', 0)),
            'appearance_in_first_grade_kinship': int(request.form.get('Appearance_in_first_grade_kinship', 0)),
            'Loss of interest in what is happening around you or doing things': int(request.form.get('Loss_of_interest_in_what_is_happening_around_you_or_doing_things', 0))
        }

        # Create complete feature set in exact order
        full_feature_order = [
            'Difficulty concentrating or staying focussed',
            'Problems remembering things that have happened recently or forgetting to do things',
            'Feeling sad low or blue',
            'Falling',
            'Difficulty getting to sleep at night or staying asleep at night',
            'Unpleasant sensations in your legs at night or while resting and a feeling that you need to move',
            'gender',
            'Finding it difficult to have sex when you try',
            'Finding it difficult to stay awake during activities such as working driving or eating',
            'Feeling that your bowel emptying is incomplete after having been to the toilet',
            'Intense vivid dreams or frightening dreams',
            'Unexplained pains (not due to known conditions such as arthritis)',
            'Excessive sweating',
            'Swelling of your legs',
            'Bowel (fecal) incontinence',
            'Constipation (less than 3 bowel movements a week) or having to strain to pass a stool (faeces)',
            'Feeling light headed dizzy or weak standing from sitting or lying',
            'appearance_in_first_grade_kinship',
            'Difficulty swallowing food or drink or problems with choking',
            'effect_of_alcohol_on_tremor',
            'A sense of urgency to pass urine makes you rush to the toilet',
            'Getting up regularly at night to pass urine',
            'Talking or moving about in your sleep as if you are acting out a dream',
            'Dribbling of saliva during the daytime',
            'Loss of interest in what is happening around you or doing things',
            'weight',
            'height',
            'age_at_diagnosis',
            'Loss or change in your ability to taste or smell'
        ]

        # Combine features maintaining order
        combined_features = {**previous_features, **new_features}
        final_df = pd.DataFrame([combined_features])[full_feature_order]

        # Make prediction
        predict_pipeline2 = PredictPipeline2()
        results2 = predict_pipeline2.predict(final_df)
        msg = "The second verification confirms a higher likelihood of Parkinson's. Please consult a doctor." if results2[0] > 0.0 else "You might have a differential disease."
    except Exception as e:
            logging.error(f"Prediction failed: {str(e)}")
            return render_template('error.html',
                                error_message="Prediction service unavailable. Please try later.")

    return render_template('final.html', results=msg)

if __name__ == "__main__":
    threading.Timer(0.25, open_browser).start()
    app.run(host="0.0.0.0", port=5000)
