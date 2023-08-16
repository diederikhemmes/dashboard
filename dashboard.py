import streamlit as st
# import plotly.express as px 
# import plotly.graph_objects as go
import pandas as pd
# import matplotlib.pyplot as plt
import datetime
# import numpy as np
from streamlit import components
from PIL import Image

import gspread
from google.oauth2 import service_account

def generate_help(options, examples):
    help_string = 'These are examples of the answers: \n\n' 
    for n, t in enumerate(options):
        help_string += '**'
        help_string += t
        help_string += '**'
        help_string += ': '
        help_string += examples[n]
        help_string += '\n\n'
    
    return help_string
    

def riskDriver(riskdriver_groupby, riskdriver, number):

    # Define column width
    col1, col2, col3 = st.columns([8, 1, 3])

    # Retrieve driver information from groupby object
    subvariables = riskdriver_groupby['Variable'].unique().tolist()
    overwrites = riskdriver_groupby['Overwrites'].unique().tolist()
    overwrites.insert(0, 'No overwrite')

    # Create drop down for riskdriver
    with col1:
        with st.expander('Risk Driver ' + str(number) + ': ' + riskdriver):
            inputs = [None]*len(subvariables)

            # Loop over variables
            for i, sv in enumerate(subvariables):
                options = riskdriver_groupby[riskdriver_groupby['Variable']==sv]['Answers'].tolist()
                examples = riskdriver_groupby[riskdriver_groupby['Variable']==sv]['Example'].tolist()
                string = generate_help(options, examples)
                choice = st.selectbox(sv, options, help=string)

                # Determine score for subvariable (sv)
                sv_score = options.index(choice) + 1
                sv_score_rel = sv_score / len(options)
                inputs[i] = sv_score_rel

            # Determince score for risk driver
            score = sum(inputs)
            score_rel = score / len(subvariables)

    # Display score for risk driver based on score per varialbe
    with col2:
        st.text_input(label='score', value=round(score_rel, 2), label_visibility='collapsed', disabled=True, key=str(number)+'col2')
    
    # Create input box for overwriting risk driver score
    with col3:
        overwrite = st.selectbox('Overwrite', overwrites, label_visibility='collapsed')
        if overwrite is not 'No overwrite':
            score = overwrites.index(overwrite) 
            score_rel = score / (len(overwrites)-1)

    return score_rel, inputs, sum(inputs), overwrite


def convert_range(value, min_original, max_original, min_new, max_new):

    # Calculate the range of the original values
    range_original = max_original - min_original

    # Scale and shift the value to the new range
    scaled_value = (value - min_original) / range_original * (max_new - min_new) + min_new

    return scaled_value


def apply_weights(weights_df, finance_weight, risk_weight, invest_weight, busdev_weight, risk_driver, value):

    # Retrieve which weights are selected
    selected_groups = []
    if finance_weight:
        selected_groups.append('Sub score Finance')
    if risk_weight:
        selected_groups.append('Sub score Risk')
    if invest_weight:
        selected_groups.append('Sub score Invest')
    if busdev_weight:
        selected_groups.append('Sub score Bus Dev')

    # Calculate the average of the selected weights
    selected_weights_df = weights_df[selected_groups]
    weights_df['Sum'] = selected_weights_df.sum(axis=1)


    total_points = weights_df['Sum'].sum()
    weights_df['Weight'] = weights_df['Sum'] / total_points

    # st.dataframe(weights_df)

    # Obtain and apply weight
    weight = weights_df.loc[weights_df['Risk Factor'] == risk_driver, 'Weight'].values[0]
    weighted_value = value * weight

    return weighted_value



if __name__ == '__main__':

    # Create logo header
    col1, col2, col3 = st.columns([2,4,2])
    image_RQ = Image.open('Images/logo_RQ_rgb.jpg')
    image_C8 = Image.open('Images/Copper8_logo.png')
    image_CF = Image.open('Images/circular_finance_lab_logo.png')
    with col2:
        st.image(image_CF)
    col1, col2, col3 = st.columns([10, 1, 7]) #st.columns([10,6.5])
    with col1:
        st.image(image_RQ)
    with col3:
        st.image(image_C8)

    # Create tabs
    tab1, tab2= st.tabs(["Scorecard", "Readme"])

#####################################################################################  
# Scorecard tab
#####################################################################################  
    with tab1:

        st.header('Fill in details')
        user = st.text_input(label='Filled in by', placeholder='Your name/company', key='name_key')
        client_company = st.text_input(label='Filled in for', placeholder='Name/company for which to determine a risk score', key='client_key')
        date = datetime.datetime.now().date()
        time = datetime.datetime.now().time().replace(microsecond=0)
        version = 'v0.61'
        row = [str(date), str(time), version, user, client_company]
        info_columns = len(row)

        # Set up the scope and credentials to Google Sheet
        credentials = service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"],
            scopes=[
                "https://www.googleapis.com/auth/spreadsheets",
            ],
        )
    
        # Open Google Sheet by its url
        client = gspread.authorize(credentials)
        sheets_url = st.secrets["sheet_url"]
        sh = client.open_by_url(sheets_url)

        # Weights selection UI
        st.header('Choose expert weights')
        col1, col2, col3, col4 = st.columns([1,1,1,1])
        with col1:
            finance_weight = st.checkbox('Finance', value=True)
        with col2:
            risk_weight = st.checkbox('Risk', value=True)
        with col3:
            invest_weight = st.checkbox('Invest', value=True)
        with col4:
            busdev_weight = st.checkbox('Bus Dev', value=True)
        if not any([finance_weight, risk_weight, invest_weight, busdev_weight]):
            st.error('Please select at least one expert weight', icon="🚨")

        # Read in expert weights file 
        weights_df = pd.read_excel('Expert_weights.xlsx')

        # Find maximum weight to use for normalization
        max_weight = weights_df.iloc[:, 1:].max().max()

        st.header('Score risk factors')

        # Read in risk driver file
        drivers_df = pd.read_excel('RD_test.xlsx')
        drivers_df = drivers_df.fillna(method='ffill', axis=0)

        # Create dropdown for each risk driver
        n = 1
        risk_scores = []
        for name, rd in drivers_df.groupby('Risk Driver', sort=False):
            risk, inputs, score, overwrite = riskDriver(rd, name, n)
            
            # Apply expert weight
            weighted_risk = apply_weights(weights_df, finance_weight, risk_weight, invest_weight, busdev_weight, name, risk)

            # Store risk driver information
            risk_scores.append(weighted_risk)
            row.extend(inputs)
            row.append(score)
            row.append(overwrite)
            n += 1

        st.header('Determine risk score')

        # Determine final score and convert in on scale 0 to 100
        final_score = sum(risk_scores)
        min_score = 0.25
        normalized_final_score = convert_range(final_score, min_original=min_score, max_original=1, min_new=0, max_new=100)

        # Display final score and insert to correct column
        col1, col2 = st.columns([3,2])
        with col1:
            st.text('Final Score: ' + str(round(final_score, 3)))
            st.text('Normalized final Score: ' + str(round(normalized_final_score, 3)))
        with col2:
            internal_score = st.number_input('Internal score', min_value=0., max_value=1., step=0.01)
        
        # Store scorecard score and internal score
        row.insert(info_columns, normalized_final_score)
        row.insert(info_columns+1, internal_score)
        st.text(finance_weight)
        row.extend([finance_weight, risk_weight, invest_weight, busdev_weight])

        # Request feedback
        feedback = st.text_area('Feedback on this form', placeholder="Please provide here any feedback that you have on, e.g., convenience of filing in this form, or how well the score from this form aligns with your internal score. ")
        row.append(feedback)

        # Submit data to google sheet
        send = st.button("Submit")
        if send:    
            st.success('Risk score submitted!')
            sh.sheet1.append_row(row)

#####################################################################################  
# Readme tab
#####################################################################################  
    with tab2:
        st.markdown('**Names**: It is possible to use dummy names for the \'filled \
                     in by\' and \'filled in for\' fields.')

        st.markdown("**Scoring risk drivers**: The scorecard contains six risk drivers, \
            each with multiple variables. If you are unsure about how to rate the \
            variables, it is possible to provide an overwrite for the risk driver \
            score. This overwrite replaces the score determined on the separate variables."
        ) 

        st.markdown("**Expert weights**: Different groups of experts distributed points over \
                    the different risk drivers. It is possible to select expert groups to weight \
                    the circular score."
        ) 


        st.markdown("**Final score**: The form will show the determined circular score,\
                     which lies between 0 (lowest) and 100 (good). Please fill in your \
                    internel Probability of Default score for comparison.")


