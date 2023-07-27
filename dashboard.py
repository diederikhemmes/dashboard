import streamlit as st
# import plotly.express as px 
# import plotly.graph_objects as go
# import pandas as pd
# import matplotlib.pyplot as plt
import datetime
# import numpy as np

import gspread
from google.oauth2 import service_account


def riskDriver(subvariables, number):

    # Define column width
    col1, col2, col3 = st.columns([6, 1, 2])
    
    # Create drop down with score per variable
    with col1:
        with st.expander('Risk Driver ' + str(number)):
            subcols = st.columns(subvariables)
            inputs = [None]*subvariables
            for i in range(subvariables):
                with subcols[i]:
                    inputs[i] = float(st.number_input('variable '+str(i+1), value=1, min_value=1, max_value=4, key=str(number)+'_'+str(i)))
            score = sum(inputs)

    # Display score for risk driver based on score per varialbe
    with col2:
        st.text_input(label='score', value=score, label_visibility='collapsed', disabled=True, key=str(number)+'col2')
    
    # Create input box for overwriting risk driver score
    with col3:
        overwrite = st.text_input(label='overwrite', placeholder='overwrite', label_visibility='collapsed', key=str(number)+'col3')
        if overwrite:
            # Check if input is numerical
            if overwrite.strip().isdigit():
                # Check if input is in the permitted range
                if float(overwrite) < 1*subvariables or float(overwrite) > 4*subvariables:
                    with col1:
                        st.error('Overwrite should be between ' + str(1*subvariables) + ' and ' + str(4*subvariables), icon="🚨")
                else:
                    score = float(overwrite)
            else:
                with col1:
                    st.error('Overwrite should be a whole number', icon="🚨")

    return score/subvariables, inputs, sum(inputs), overwrite

def convert_range(value):
    # Define the original range
    min_original = 0.25
    max_original = 1.0

    # Define the new range
    min_new = 0.0
    max_new = 1.0

    # Calculate the range of the original values
    range_original = max_original - min_original

    # Scale and shift the value to the new range
    scaled_value = (value - min_original) / range_original * (max_new - min_new) + min_new

    return scaled_value


if __name__ == '__main__':

    st.header('Fill in details')
    user = st.text_input(label='Filled in by', placeholder='Your name/company', key='name_key')
    client_company = st.text_input(label='Filled in for', placeholder='Name/company for which to determine a risk score', key='client_key')
    date = datetime.datetime.now().date()
    version = 'v0.1'
    row = [str(date), version, user, client_company]

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
        st.checkbox('Finance')
    with col2:
        st.checkbox('Risk')
    with col3:
        st.checkbox('Invest')
    with col4:
        st.checkbox('Bus Dev')

    st.header('Score risk factors')

    # RD 1
    # TODO denk dict van variables en dan hierover loopen (als t kan, moet variabelen hardcoden denk ik)
    number = 1
    subvariables = 4
    risk_1, inputs_1, score_1, overwrite_1 = riskDriver(subvariables, number)
    row.extend(inputs_1)
    row.append(score_1)
    row.append(overwrite_1)

    # RD 1
    number = 2
    subvariables = 3
    risk_2, inputs_2, score_2, overwrite_2 = riskDriver(subvariables, number)
    # row.extend(inputs_2)

    st.header('Determine risk score')

    # Normalize risk score based on amount of risk drivers
    number_of_RiskDrivers = 2
    total_score = (risk_1 + risk_2)/number_of_RiskDrivers
    inv_total_score = 1/total_score
    final_score = convert_range(inv_total_score)

    # Display final score and insert to correct column
    st.text('Final Score: ' + str(final_score))
    row.insert(4, final_score)
    st.text('hoi')
    # Submit data to google sheet
    send = st.button("Submit")
    if send:    
        st.success('Risk score submitted!')
        sh.sheet1.append_row(row)
