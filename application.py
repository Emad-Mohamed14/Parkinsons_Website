import pickle
from flask import Flask,request,render_template, url_for
import numpy as np
import pandas as pd
from src.logger import logging
import webbrowser
import threading

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline1 import CustomData,PredictPipeline

application = Flask(__name__)
app = application

#Route for a Home Page

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

@app.route('/predictdata1',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('prediction.html',results=None)
    else:
        data=CustomData(
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
        if(results[0]==0.0):
            msg = "No"
        else:
            msg = "Yes"
        logging.info(f"Predicted result: {results[0]}, Message: {msg}")
        return render_template('post.html',results=msg)


if __name__=="__main__":
    threading.Timer(1.25, open_browser).start()
    app.run(host="0.0.0.0", port=5000)
