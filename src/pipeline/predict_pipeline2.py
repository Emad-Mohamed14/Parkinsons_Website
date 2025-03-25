import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import joblib

class PredictPipeline2:
    def predict(self, features):
        try:
            model_path = os.path.join("artifacts","model2.pkl")
            preprocessor_path = os.path.join('artifacts','preprocessor2.pkl')
            
             # Load model and features as tuple
            model, feature_names = load_object(file_path=model_path)  # Changed here
            preprocessor = load_object(file_path=preprocessor_path)

            # Ensure correct column order
            required_columns = [
                'Difficulty concentrating or staying focussed',
                'Problems remembering things that have happened recently or forgetting to do things',
                'Feeling sad low or blue', 'Falling', 'Difficulty getting to sleep at night or staying asleep at night', 'Unpleasant sensations in your legs at night or while resting and a feeling that you need to move', 'gender', 'Finding it difficult to have sex when you try', 'Finding it difficult to stay awake during activities such as working driving or eating', 'Feeling that your bowel emptying is incomplete after having been to the toilet', 'Intense vivid dreams or frightening dreams', 'Unexplained pains (not due to known conditions such as arthritis)', 'Excessive sweating', 'Swelling of your legs', 'Bowel (fecal) incontinence', 'Constipation (less than 3 bowel movements a week) or having to strain to pass a stool (faeces)', 'Feeling light headed dizzy or weak standing from sitting or lying', 'appearance_in_first_grade_kinship', 'Difficulty swallowing food or drink or problems with choking', 'effect_of_alcohol_on_tremor', 'A sense of urgency to pass urine makes you rush to the toilet', 'Getting up regularly at night to pass urine', 'Talking or moving about in your sleep as if you are acting out a dream', 'Dribbling of saliva during the daytime', 'Loss of interest in what is happening around you or doing things', 'weight', 'height', 'age_at_diagnosis', 'Loss or change in your ability to taste or smell'
            ]
            
            # Convert features to DataFrame with EXACT column names
            features = pd.DataFrame(features)[required_columns]
            
            data_scaled = preprocessor.transform(features)
            return model.predict(data_scaled)
        
        except Exception as e:
            raise CustomException(e, sys)
        

class CustomData2:
    def __init__(self,
    Unpleasant_sensations_in_legs_at_night_or_while_resting_and_a_feeling_that_you_need_to_move: int,  # Unpleasant sensations in your legs at night or while resting, and a feeling that you need to move
    Finding_it_difficult_to_have_sex_when_you_try: int,  # Finding it difficult to have sex when you try
    Difficulty_getting_to_sleep_at_night_or_staying_asleep_at_night: int,  # Difficulty getting to sleep at night or staying asleep at night
    Feeling_sad_low_or_blue: int,  # Feeling sad, low or blue
    Intense_vivid_dreams_or_frightening_dreams: int,  # Intense, vivid dreams or frightening dreams
    Unexplained_pains_not_due_to_known_conditions_such_as_arthritis: int,  # Unexplained pains (not due to known conditions such as arthritis)
    Difficulty_concentrating_or_staying_focused: int,  # Difficulty concentrating or staying focussed
    Excessive_sweating: int,  # Excessive sweating
    Finding_it_difficult_to_stay_awake_during_activities_such_as_working_driving_or_eating: int,  # Finding it difficult to stay awake during activities such as working, driving or eating
    Feeling_that_your_bowel_emptying_is_incomplete_after_having_been_to_the_toilet: int,  # Feeling that your bowel emptying is incomplete after having been to the toilet
    Swelling_of_your_legs: int,  # Swelling of your legs
    Bowel_fecal_incontinence: int,  # Bowel (fecal) incontinence
    Appearance_in_first_grade_kinship: int,  # appearance_in_first_grade_kinship
    Feeling_light_headed_dizzy_or_weak_standing_from_sitting_or_lying: int,  # Feeling light headed, dizzy or weak standing from sitting or lying
    Loss_of_interest_in_what_is_happening_around_you_or_doing_things: int  # Loss of interest in what is happening around you or doing things
    ):
        
        self.Unpleasant_sensations_in_legs_at_night_or_while_resting_and_a_feeling_that_you_need_to_move = Unpleasant_sensations_in_legs_at_night_or_while_resting_and_a_feeling_that_you_need_to_move  
        self.Finding_it_difficult_to_have_sex_when_you_try = Finding_it_difficult_to_have_sex_when_you_try  
        self.Difficulty_getting_to_sleep_at_night_or_staying_asleep_at_night = Difficulty_getting_to_sleep_at_night_or_staying_asleep_at_night  
        self.Feeling_sad_low_or_blue = Feeling_sad_low_or_blue  
        self.Intense_vivid_dreams_or_frightening_dreams = Intense_vivid_dreams_or_frightening_dreams  
        self.Unexplained_pains_not_due_to_known_conditions_such_as_arthritis = Unexplained_pains_not_due_to_known_conditions_such_as_arthritis  
        self.Difficulty_concentrating_or_staying_focused = Difficulty_concentrating_or_staying_focused  
        self.Excessive_sweating = Excessive_sweating  
        self.Finding_it_difficult_to_stay_awake_during_activities_such_as_working_driving_or_eating = Finding_it_difficult_to_stay_awake_during_activities_such_as_working_driving_or_eating  
        self.Feeling_that_your_bowel_emptying_is_incomplete_after_having_been_to_the_toilet = Feeling_that_your_bowel_emptying_is_incomplete_after_having_been_to_the_toilet  
        self.Swelling_of_your_legs = Swelling_of_your_legs  
        self.Bowel_fecal_incontinence = Bowel_fecal_incontinence  
        self.Appearance_in_first_grade_kinship = Appearance_in_first_grade_kinship  
        self.Feeling_light_headed_dizzy_or_weak_standing_from_sitting_or_lying = Feeling_light_headed_dizzy_or_weak_standing_from_sitting_or_lying  
        self.Loss_of_interest_in_what_is_happening_around_you_or_doing_things = Loss_of_interest_in_what_is_happening_around_you_or_doing_things  

        
    def get_data_as_data_frame2(self):
        try:
            custom_data_input_dict = {
                "Unpleasant sensations in your legs at night or while resting, and a feeling that you need to move": [self.Unpleasant_sensations_in_legs_at_night_or_while_resting_and_a_feeling_that_you_need_to_move],
                "Finding it difficult to have sex when you try": [self.Finding_it_difficult_to_have_sex_when_you_try],
                "Difficulty getting to sleep at night or staying asleep at night": [self.Difficulty_getting_to_sleep_at_night_or_staying_asleep_at_night],
                "Feeling sad, low or blue": [self.Feeling_sad_low_or_blue],
                "Intense, vivid dreams or frightening dreams": [self.Intense_vivid_dreams_or_frightening_dreams],
                "Unexplained pains (not due to known conditions such as arthritis)": [self.Unexplained_pains_not_due_to_known_conditions_such_as_arthritis],
                "Difficulty concentrating or staying focussed": [self.Difficulty_concentrating_or_staying_focused],
                "Excessive sweating": [self.Excessive_sweating],
                "Finding it difficult to stay awake during activities such as working, driving or eating": [self.Finding_it_difficult_to_stay_awake_during_activities_such_as_working_driving_or_eating],
                "Feeling that your bowel emptying is incomplete after having been to the toilet": [self.Feeling_that_your_bowel_emptying_is_incomplete_after_having_been_to_the_toilet],
                "Swelling of your legs": [self.Swelling_of_your_legs],
                "Bowel (fecal) incontinence": [self.Bowel_fecal_incontinence],
                "appearance_in_first_grade_kinship": [self.Appearance_in_first_grade_kinship],
                "Feeling light headed, dizzy or weak standing from sitting or lying": [self.Feeling_light_headed_dizzy_or_weak_standing_from_sitting_or_lying],
                "Loss of interest in what is happening around you or doing things": [self.Loss_of_interest_in_what_is_happening_around_you_or_doing_things]
            }

            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e, sys)