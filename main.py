import streamlit as st
import pickle
import pandas as pd
import numpy as np
import xgboost
from xgboost import XGBRegressor

pipe = pickle.load(open('newmodel.pkl', 'rb'))

st.title('Khelo Dimag Se')
st.sidebar.title('  Cricket Score Predictor')

from PIL import Image

image = Image.open('C:/Users/AMIT/Downloads/crick1.jpg')

st.image(image)

teams = ['Australia',
         'India',
         'Bangladesh',
         'New Zealand',
         'South Africa',
         'England',
         'West Indies',
         'Afghanistan',
         'Pakistan',
         'Sri Lanka']

cities = ['Colombo',
          'Mirpur',
          'Johannesburg',
          'Dubai',
          'Auckland',
          'Cape Town',
          'London',
          'Pallekele',
          'Barbados',
          'Sydney',
          'Melbourne',
          'Durban',
          'St Lucia',
          'Wellington',
          'Lauderhill',
          'Hamilton',
          'Centurion',
          'Manchester',
          'Abu Dhabi',
          'Mumbai',
          'Nottingham',
          'Southampton',
          'Mount Maunganui',
          'Chittagong',
          'Kolkata',
          'Lahore',
          'Delhi',
          'Nagpur',
          'Chandigarh',
          'Adelaide',
          'Bangalore',
          'St Kitts',
          'Cardiff',
          'Christchurch',
          'Trinidad']

col1, col2 = st.columns(2)

with col1:
    batting_team = teams.index(st.selectbox('Select batting team', sorted(teams)))
with col2:
    bowling_team = teams.index(st.selectbox('Select bowling team', sorted(teams)))

city = cities.index(st.selectbox('Select city', sorted(cities)))

col3, col4, col5 = st.columns(3)

with col3:
    current_score = st.number_input('Current Score')
with col4:
    overs = st.number_input('Overs done(works for over>5)')
with col5:
    wickets = st.number_input('Wickets out')

last_five = st.number_input('Runs scored in last 5 overs')

if st.button('Predict Score'):
    balls_left = 120 - (overs * 6)
    wickets_left = 10 - wickets
    crr = current_score / overs

    input_df = pd.DataFrame(
        {'current_score': [current_score], 'balls_left': [balls_left], 'wickets_left': [wickets], 'crr': [crr],
         'last_five': [last_five]})
    result = pipe.predict(np.array([[batting_team, bowling_team, city, current_score, overs, wickets, crr, last_five]]))[0]
    st.write(result)
    st.header("Predicted Score - " + str(int(result)))



import streamlit as st
from streamlit_option_menu import option_menu

# 1=sidebar menu, 2=horizontal menu, 3=horizontal menu w/ custom menu
EXAMPLE_NO = 1


def streamlit_menu(example=1):
    if example == 1:
        # 1. as sidebar menu
        with st.sidebar:
            selected = option_menu(
                menu_title="Main Menu",  # required
                options=["Home", "Projects", "Settings"],  # required
                icons=["house", "book", "gear"],  # optional
                menu_icon="cast",  # optional
                default_index=0,  # optional
            )
        return selected

    if example == 2:
        # 2. horizontal menu w/o custom style
        selected = option_menu(
            menu_title=None,  # required
            options=["Home", "Projects", "Settings"],  # required
            icons=["house", "book", "gear"],  # optional
            menu_icon="cast",  # optional
            default_index=0,  # optional
            orientation="horizontal",
        )
        return selected

    if example == 3:
        # 2. horizontal menu with custom style
        selected = option_menu(
            menu_title=None,  # required
            options=["Home", "Projects", "Settings"],  # required
            icons=["house", "book", "gear"],  # optional
            menu_icon="cast",  # optional
            default_index=0,  # optional
            orientation="horizontal",
            styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": "orange", "font-size": "25px"},
                "nav-link": {
                    "font-size": "25px",
                    "text-align": "left",
                    "margin": "0px",
                    "--hover-color": "#eee",
                },
                "nav-link-selected": {"background-color": "green"},
            },
        )
        return selected


selected = streamlit_menu(example=EXAMPLE_NO)

