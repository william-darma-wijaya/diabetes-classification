import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load("savedPickle/model.pkl")

# Load encoder
genderEncoder = joblib.load("savedPickle/genderEncoder.pkl")
familyHistoryEncoder = joblib.load("savedPickle/familyHistoryEncoder.pkl")
favcEncoder = joblib.load("savedPickle/favcEncoder.pkl")
caecEncoder = joblib.load("savedPickle/caecEncoder.pkl")
smokeEncoder = joblib.load("savedPickle/smokeEncoder.pkl")
sccEncoder = joblib.load("savedPickle/sccEncoder.pkl")
calcEncoder = joblib.load("savedPickle/calcEncoder.pkl")
mtransEncoder = joblib.load("savedPickle/mtransEncoder.pkl")
targetEncoder = joblib.load("savedPickle/targetEncoder.pkl")

# Load scalers
ageScaler = joblib.load("savedPickle/ageScaler.pkl")
heightScaler = joblib.load("savedPickle/heightScaler.pkl")
weightScaler = joblib.load("savedPickle/weightScaler.pkl")

def split_x_y(data, target_column="NObeyesdad"):
  output_df = data[target_column]
  input_df = data.drop(target_column, axis=1)
  return input_df, output_df

def convert_input_to_df(input_data):
  data = [input_data]
  df = pd.DataFrame(data, columns = ['Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight', 'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE', 'CALC', 'MTRANS'])
  return df

def main():
  st.title('Diabetes Classification')
  st.info("This app use machine learning to classify diabetes levels.")

  st.subheader("Dataset")
  df = pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv")
  x, y = split_x_y(df)
  with st.expander("**Raw Data**"):
    st.dataframe(df.head(50))

  with st.expander("**Input Data**"):
    st.dataframe(x.head(50))

  with st.expander("**Output Data**"):
    st.dataframe(y.head(50))

  st.subheader("Height vs Weight With Obesity Level")
  with st.expander('**Data Visualization**'):
    st.scatter_chart(data=df, x = 'Height', y = 'Weight', color='NObeyesdad')

  # input data by user
  st.subheader("Input Patient Data")
  Age = st.slider('Age', min_value = 10, max_value = 65, value = 25)
  Height = st.slider('Height', min_value = 1.45, max_value = 2.00, value = 1.75)
  Weight = st.slider('Weight', min_value = 30, max_value = 180, value = 70)
  FCVC = st.slider('FCVC', min_value = 1, max_value = 3, value = 2)
  NCP = st.slider('NCP', min_value = 1, max_value = 4, value = 3)
  CH2O = st.slider('CH2O', min_value = 1, max_value = 3, value = 2)
  FAF = st.slider('FAF', min_value = 0, max_value = 3, value = 1)
  TUE = st.slider('TUE', min_value = 0, max_value = 2, value = 1)
  
  Gender = st.selectbox('Gender', ('Male', 'Female'))
  family_history_with_overweight = st.selectbox('Family history with overweight', ('yes', 'no'))
  FAVC = st.selectbox('FAVC', ('yes', 'no'))
  CAEC = st.selectbox('CAEC', ('Sometimes', 'Frequently', 'Always', 'no'))
  SMOKE = st.selectbox('SMOKE', ('yes', 'no'))
  SCC = st.selectbox('SCC', ('yes', 'no'))
  CALC = st.selectbox('CALC', ('Sometimes', 'no', 'Frequently', 'Always'))
  MTRANS = st.selectbox('MTRANS', ('Public_Transportation', 'Automobile', 'Walking', 'Motorbike', 'Bike'))

  input_data = [Gender, Age, Height, Weight, family_history_with_overweight, FAVC, FCVC, NCP, CAEC, SMOKE, CH2O, SCC, FAF, TUE, CALC, MTRANS]

  user_df = convert_input_to_df(input_data)

  st.subheader("Inputted Patient Data")
  st.dataframe(user_df)

  

  # Input data for program
  user_input = [erythema, scaling, definite_borders, itching, koebner_phenomenon, polygonal_papules, follicular_papules, oral_mucosal_involvement, knee_and_elbow_involvement, scalp_involvement, family_history, melanin_incontinence, eosinophils_infiltrate, PNL_infiltrate, fibrosis_papillary_dermis, exocytosis, acanthosis, hyperkeratosis, parakeratosis, clubbing_rete_ridges, elongation_rete_ridges, thinning_suprapapillary_epidermis, spongiform_pustule, munro_microabcess, focal_hypergranulosis, disappearance_granular_layer, vacuolisation_damage_basal_layer, spongiosis, saw_tooth_appearance_retes, follicular_horn_plug, perifollicular_parakeratosis, inflammatory_mononuclear_infiltrate, band_like_infiltrate, age]
  model_filename = "trained_model.pkl"
  model = load_model(model_filename)
  prediction = predict_with_model(model, user_input)
  st.write("The prediction output is: ", prediction)

if __name__ == "__main__":
  main()
