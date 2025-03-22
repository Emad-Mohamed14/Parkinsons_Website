import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import joblib

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path = os.path.join("artifacts","model1.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor1.pkl')
            model, feature_names = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            # data_scaled = preprocessor.transform(features)

            # Ensure only the expected features are passed to the model
            features = features[feature_names]
            data_scaled = preprocessor.transform(features)

            preds = model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e , sys)
        

class CustomData:
    def __init__(self,
        Dribbling_of_saliva_during_the_daytime: int,
        height: int,
        Falling: int,
        Memory_problems: int,
        Loss_of_taste_smell: int,
        Effect_of_alcohol: int,
        Acting_out_dreams: int,
        Difficulty_swallowing: int,
        Constipation: int,
        gender: int,
        weight: int,
        Night_urination: int,
        Urgency_to_urinate: int,
        age_at_diagnosis: int):
        
        self.Dribbling_of_saliva_during_the_daytime = Dribbling_of_saliva_during_the_daytime
        self.height = height
        self.Falling = Falling
        self.Memory_problems = Memory_problems
        self.Loss_of_taste_smell = Loss_of_taste_smell
        self.Effect_of_alcohol = Effect_of_alcohol
        self.Acting_out_dreams = Acting_out_dreams
        self.Difficulty_swallowing = Difficulty_swallowing
        self.Constipation = Constipation
        self.gender = gender
        self.weight = weight
        self.Night_urination = Night_urination
        self.Urgency_to_urinate = Urgency_to_urinate
        self.age_at_diagnosis = age_at_diagnosis
        
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Dribbling of saliva during the daytime": [self.Dribbling_of_saliva_during_the_daytime],
                "height": [self.height],
                "Falling": [self.Falling],
                "Problems remembering things that have happened recently or forgetting to do things": [self.Memory_problems],
                "Loss or change in your ability to taste or smell": [self.Loss_of_taste_smell],
                "effect_of_alcohol_on_tremor": [self.Effect_of_alcohol],
                "Talking or moving about in your sleep as if you are acting out a dream": [self.Acting_out_dreams],
                "Difficulty swallowing food or drink or problems with choking": [self.Difficulty_swallowing],
                "Constipation (less than 3 bowel movements a week) or having to strain to pass a stool (faeces)": [self.Constipation],
                "gender": [self.gender],
                "weight": [self.weight],
                "Getting up regularly at night to pass urine": [self.Night_urination],
                "A sense of urgency to pass urine makes you rush to the toilet": [self.Urgency_to_urinate],
                "age_at_diagnosis": [self.age_at_diagnosis]
            }

            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e, sys)

