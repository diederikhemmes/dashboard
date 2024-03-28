# -----------------------------------------------------------------------------
# Title: Circularity Score Dashboard - Insights
# Description: This script creates a streamlit dashboard.
# -----------------------------------------------------------------------------

import streamlit as st
import pandas as pd
import datetime
import pytz 
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import colormaps
import numpy as np
import csv
import seaborn as sns
import altair as alt
import warnings
warnings.filterwarnings("ignore")

@st.cache_data
def load_images(image_path: str) -> None:
    """
    Load and display images in a formatted layout.

    This function loads and displays images from provided file paths in a
    structured layout using Streamlit's column layout features.

    Args:
        image_path (str): File path to the image.

    Returns:
        None
    """
    logos = Image.open(image_path)
    st.image(logos, width=400)

    return

import streamlit as st
import streamlit_authenticator as stauth

names = st.secrets['streamlit_login']['names']
usernames = st.secrets['streamlit_login']['usernames']
passwords = st.secrets['streamlit_login']['passwords']

credentials = {"usernames":{}}
        
for uname,name,pwd in zip(usernames,names,passwords):
    user_dict = {"name": name, "password": pwd}
    credentials["usernames"].update({uname: user_dict})

# hashed_passwords = stauth.hasher(passwords).generate()
authenticator = stauth.Authenticate(credentials, 'circular_risk', 'risk_sign', cookie_expiry_days=1)

name, authentication_status, username = authenticator.login()

# if authentication_status == False:
#     st.error('Username/password is incorrect')

# if authentication_status == None:
#     st.error('Please enter your username and password')

if st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')

# if authentication_status == True:
elif st.session_state["authentication_status"]:
    authenticator.logout()
    st.write(f'Welcome *{st.session_state["name"]}*')
    st.title('Circular Risk Scorecard - Insights')
    st.success('You are logged in!')
    #### main ##################################################################################################
    if __name__ == '__main__':
        
        version = '0.1'

        st.subheader("version: " + version)

        exclude = st.checkbox("Exclude test data")

        if exclude:
            # Read in latest data file 
            filepath = 'Outlook_data.csv'
            data = pd.read_csv(filepath)
            data = data.loc[data['test_run'] == False]
            records = data.reset_index(drop=True)
            
        else:
            # Read in latest data file 
            filepath = 'Outlook_data.csv'
            records = pd.read_csv(filepath)

        # Create dashboard tabs
        tab1, tab2 = st.tabs(["Summary", "Raw Data"])
        
        # Obtain general information
        date = datetime.datetime.now(tz=pytz.timezone('Europe/Amsterdam')).date()
        time = datetime.datetime.now(tz=pytz.timezone('Europe/Amsterdam')).time().replace(microsecond=0)
        

        # list of business models
        business_models = ['Product as a Service', 'Resource recovery (material sales model)', 'Circular supplies (product sales model)', 'Product life-time extension (service sales model)', 'Sharing platforms', 'Other']


    #####################################################################################  
        # Insights tab
    #####################################################################################
    
        with tab1:

            # Add some space
            ''
            st.subheader('Condensed data', divider='red')

            # Display the summary of the latest data
            summary_columns = ['Date', 'Institution', 'User', 'Client_company', 'BusinessModel', 'Final_score_normalized']
            st.dataframe(records[summary_columns])
            # st.write(records.describe())
            # ---------------------------------------------------------------------------------------

            # Add some space
            ''
            st.subheader('Average scores accross business models', divider='red')

            # st.bar_chart(records, x = 'BusinessModel', y = 'Final_score_normalized')
            df = records.groupby(["BusinessModel"])['Final_score_normalized'].mean().reset_index(name='Average score')
            # st.bar_chart(df, x= 'BusinessModel', y = 'Average score')

            # using Altair
            chart = alt.Chart(df).mark_bar().encode(
                x=alt.X('BusinessModel', title='Business Model'),
                y=alt.Y('Average score', title='Average score')
            ).configure_axis(
                labelAngle=0
            ).interactive()
            st.altair_chart(chart, use_container_width=True)

            
            # ---------------------------------------------------------------------------------------

            # Add some space
            ''
            st.subheader('Distribution of different business models', divider='red')

            # pie chart
            plt.pie(records['BusinessModel'].value_counts(), labels=records['BusinessModel'].value_counts().index, autopct='%1.1f%%', startangle=140, colors = plt.cm.Set2.colors)
            plt.axis('equal')
            fig = plt.gcf()
            st.pyplot(fig)

            
            # ---------------------------------------------------------------------------------------
            ''
            
            st.subheader('Histogram of Final Scores', divider='red')
            # histogram
            chart = alt.Chart(records).mark_bar().encode(
                alt.X('Final_score_normalized', bin=alt.Bin(maxbins=20), title='Final Score'),
                y='count()',
            ).properties(
                width=600,
                height=400
                # title='Histogram of Final Scores'
            )

            st.altair_chart(chart, use_container_width=True)

            # ---------------------------------------------------------------------------------------

            st.subheader('Number of submissions by Date and User', divider='red')

            dates = records['Date']
            users = records['User']
            data_temp = {'Date': dates, 'Users': users}
            users_data = pd.DataFrame(data_temp)
            users_data['Users'] = users_data['Users'].fillna('anonymous')

            # number of entries per day and user
            submissions = users_data.groupby(['Date', 'Users'], dropna=False).size().unstack(fill_value=0)
            submissions = submissions.reset_index()

            # Melt the DataFrame for Altair
            melted_df = pd.melt(submissions, id_vars=['Date'], var_name='Users', value_name='Number of Submissions')

            chart = alt.Chart(melted_df).mark_bar().encode(
                x=alt.X('Date:T', title='Date'),
                y=alt.Y('Number of Submissions:Q', title='Number of Submissions'),
                color=alt.Color('Users:N', title='Users'),
                tooltip=['Date:T', 'Number of Submissions:Q', 'Users:N']
            ).properties(
                width=600,
                height=400,
                # title='Number of Submissions by Date and User'
            # ).configure_axis(
            #     labelAngle=45
            ) #.interactive()

            st.altair_chart(chart, use_container_width=True)


            # ---------------------------------------------------------------------------------------
            # Add some space
            ''
            st.subheader('Dashboard usage', divider='red')

            date_counts = records.groupby('Date').size().reset_index(name='Number of entries')
            st.line_chart(date_counts, x= 'Date', y = 'Number of entries')

            # line chart using Altair
            # chart = alt.Chart(date_counts).mark_line().encode(
            #     x='Date:T',
            #     y='Number of entries',
            #     tooltip=['Date:T', 'Number of entries:Q']
            # ).properties(
            #     width=600,
            #     height=400
            # )

            # st.altair_chart(chart, use_container_width=True)



    #####################################################################################  
        # Raw data tab
    #####################################################################################
        with tab2:
            # Display the latest data
            st.dataframe(records)
                


        # Add some space
        ''
        ''
        ''
        ''
        # Create logo header
        st.text(' ')
        st.text(' ')
        st.subheader('This dashboard was realized by:')
        image_path = 'Images/Logos4.png'
        load_images(image_path)



