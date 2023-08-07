import streamlit as st
# import plotly.express as px 
# import plotly.graph_objects as go
import pandas as pd
# import matplotlib.pyplot as plt
import datetime
# import numpy as np
from streamlit import components

import gspread
from google.oauth2 import service_account

def riskDriver(riskdriver, variables, number):

    # Define column width
    col1, col2, col3 = st.columns([6, 1, 2])

    subvariables = len(variables)
    
    # Create drop down with score per variable
    with col1:
        with st.expander('Risk Driver ' + str(number) + ': ' + riskdriver):
            subcols = st.columns(subvariables)
            inputs = [None]*subvariables
            for i in range(subvariables):
                with subcols[i]:
                    inputs[i] = float(st.number_input('variable '+str(i+1) + ': ' + variables[i], value=1, min_value=1, max_value=4, key=str(number)+'_'+str(i)))
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

def generate_help(options):

    # TODO read out examples
    examples = ['No diversity in Management Team on e.g. Ethnical background, Gender, Type of working experience (e.g. start-up, scale-up, corporate, institution) and Personality types (green/red/yellow/blue)', ' example 2', 'example 3', ' example 4']

    help_string = 'These are examples of the answers: \n\n' 
    for n, t in enumerate(options):
        help_string += t
        help_string += ': '
        help_string += examples[n]
        help_string += '\n\n'
    
    return help_string
    

def riskDriver2(riskdriver_groupby, riskdriver, number):
    # Define column width
    col1, col2, col3 = st.columns([6, 1, 2])

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
                string = generate_help(options)
                choice = st.selectbox(sv, options, help=string)

                examples = ['example 1', ' example 2', 'example 3', ' example 4']
                st.markdown(examples[options.index(choice)])

                # Determine score for subvariable
                sv_score = options.index(choice) + 1
                sv_score_rel = sv_score / len(options)
                inputs[i] = sv_score_rel

            # Determince score for risk driver
            score = sum(inputs)
            score_rel = score / len(subvariables)

    # Display score for risk driver based on score per varialbe
    with col2:
        st.text_input(label='score', value=score_rel, label_visibility='collapsed', disabled=True, key=str(number)+'col2')
    
    # Create input box for overwriting risk driver score
    with col3:
        overwrite = st.selectbox('Overwrite', overwrites, label_visibility='collapsed')
        if overwrite is not 'No overwrite':
            score = overwrites.index(overwrite) 

    return score_rel, inputs, sum(inputs), overwrite


def convert_range(value, min_original, max_original, min_new, max_new):

    # Calculate the range of the original values
    range_original = max_original - min_original

    # Scale and shift the value to the new range
    scaled_value = (value - min_original) / range_original * (max_new - min_new) + min_new

    return scaled_value


def apply_weights(weights_df, finance_weight, risk_weight, invest_weight, busdev_weight, risk_driver, value):
     # TODO misschien later in 1x applyen ipv per risk driver? 

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
    weights_df['Average'] = selected_weights_df.mean(axis=1)

    # st.dataframe(weights_df)

    # Obtain and apply weight
    weight = weights_df.loc[weights_df['Risk Factor'] == risk_driver, 'Average'].values[0]
    weighted_value = value * weight

    return weighted_value



if __name__ == '__main__':

    st.header('Fill in details')
    user = st.text_input(label='Filled in by', placeholder='Your name/company', key='name_key')
    client_company = st.text_input(label='Filled in for', placeholder='Name/company for which to determine a risk score', key='client_key')
    date = datetime.datetime.now().date()
    time = datetime.datetime.now().time().replace(microsecond=0)
    version = 'v0.1'
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
    # st.dataframe(drivers)

    # Create dropdown for each risk driver
    n = 1
    for name, rd in drivers_df.groupby('Risk Driver', sort=False):
        riskDriver2(rd, name, n)
        n += 1

    # RD 1
    st.text('old:')
    # TODO denk dict van variables en dan hierover loopen (als t kan, moet variabelen hardcoden denk ik)
    riskdriver_1 = 'Resource availability'
    subvariables_1 = ['Dependency on Critical Raw Materials', 'Closest peak year of critical raw materials', 'Ownership/control over resources (natural hedge)', 'Type of relationship with value chain']
    number = n 
    risk_1, inputs_1, score_1, overwrite_1 = riskDriver(riskdriver_1, subvariables_1, number)
    row.extend(inputs_1)
    row.append(score_1)
    row.append(overwrite_1)
    inversed_risk_1 = 1/risk_1
    scaled_risk_1 = convert_range(inversed_risk_1, min_original=0.25, max_original=1.0, min_new=0.0, max_new=1.0)
    weighted_risk_1 = apply_weights(weights_df, finance_weight, risk_weight, invest_weight, busdev_weight, riskdriver_1, scaled_risk_1)

    # RD 2
    riskdriver_2 = 'Circularity of asset'
    subvariables_2 = ['xx', 'yy', 'zz']
    number = n + 1
    risk_2, inputs_2, score_2, overwrite_2 = riskDriver(riskdriver_2, subvariables_2, number)
    inversed_risk_2 = 1/risk_2
    scaled_risk_2 = convert_range(inversed_risk_2, min_original=0.25, max_original=1.0, min_new=0.0, max_new=1.0)
    weighted_risk_2 = apply_weights(weights_df, finance_weight, risk_weight, invest_weight, busdev_weight, riskdriver_2, scaled_risk_2)

    st.header('Determine risk score')

    # Normalize risk score based on amount of risk drivers
    number_of_RiskDrivers = 2
    final_score = (weighted_risk_1 + weighted_risk_2)/number_of_RiskDrivers
    normalized_final_score = convert_range(final_score, min_original=0, max_original=max_weight, min_new=0, max_new=100)

    # Display final score and insert to correct column
    st.text('Final Score: ' + str(normalized_final_score))
    row.insert(info_columns, normalized_final_score)

    # Submit data to google sheet
    send = st.button("Submit")
    if send:    
        st.success('Risk score submitted!')
        sh.sheet1.append_row(row)
