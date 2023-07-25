import streamlit as st
# import plotly.express as px 
# import plotly.graph_objects as go
import pandas as pd
# import matplotlib.pyplot as plt

import gspread
from google.oauth2 import service_account



def riskDriver(subvariables, number):
    col1, col2, col3 = st.columns([6, 1, 2])
    
    with col1:
        with st.expander('Risk Driver ' + str(number)):
            subcols = st.columns(subvariables)
            inputs = [None]*subvariables
            for i in range(subvariables):
                with subcols[i]:
                    inputs[i] = float(st.number_input('variable '+str(i+1), value=1, min_value=1, max_value=4, key=str(number)+'_'+str(i)))
            score = sum(inputs)

    with col2:
        st.text_input(label='score', value=score, label_visibility='collapsed', disabled=True, key=str(number)+'col2')
    
    with col3:
        overwrite = st.text_input(label='overwrite', placeholder='overwrite', label_visibility='collapsed', key=str(number)+'col3')
        if overwrite:
            if float(overwrite) < 1*subvariables or float(overwrite) > 4*subvariables:
                with col1:
                    st.error('Overwrite should be between ' + str(1*subvariables) + ' and ' + str(4*subvariables), icon="🚨")
            else:
                score = float(overwrite)


    return score / subvariables
    

if __name__ == '__main__':

    st.write('hoi')

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
    number = 1
    subvariables = 4
    score_1 = riskDriver(subvariables, number)

    # RD 1
    number = 2
    subvariables = 3
    score_2 = riskDriver(subvariables, number)

    st.header('Determine risk score')

    # Normalize risk score based on amount of risk drivers
    number_of_RiskDrivers = 2
    total_score = (score_1 + score_2)/number_of_RiskDrivers
    inv_total_score = 1/total_score

    st.text('Final Score: ' + str(inv_total_score))

    send = st.button("Submit")
    if send:    
        sh.sheet1.append_row([inv_total_score])
        st.success('Risk score submitted!')


    
# # st.header('Layout option 1: ')
# col1, col2, col3 = st.columns([5, 1, 1])
# with col1:
#     with st.expander('Risk Driver 1'):
#         subcols = st.columns(4)
#         inputs = [None]*4
#         for i in range(4):
#             with subcols[i]:
#                 inputs[i] = float(st.number_input('variable '+str(i+1), value=1, min_value=1, max_value=4, key=i))
#         score = sum(inputs)
# with col2:
#     # a = st.text_input(label='score', placeholder='score', label_visibility='collapsed')
#     a = st.text_input(label='score', value=score, label_visibility='collapsed', disabled=True)

#     # st.write(str(score))
# with col3:
#     a = st.text_input(label='overwrite', placeholder='overwrite', label_visibility='collapsed')