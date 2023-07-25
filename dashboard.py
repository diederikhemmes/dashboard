import streamlit as st
# import plotly.express as px 
# import plotly.graph_objects as go
import pandas as pd
# import matplotlib.pyplot as plt

# import gspread
from google.oauth2 import service_account
from gsheetsdb import connect



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


    return score


# def write_data(sheets_url):
    # csv_url = sheets_url.replace("/edit#gid=", "/export?format=csv&gid=")
    # sht2 = gc.open_by_url(sheets_url)
    # sht2.update('B1', 'Bingo!')


    # existing_data = pd.read_csv(csv_url)
    # with open(csv_url, 'a') as outfile:
    #     outfile.write('hoi')

    

def run_query(query):
    rows = conn.execute(query, headers=1)
    rows = rows.fetchall()
    return rows

if __name__ == '__main__':

    # print(' hoi')
    # gc = gspread.service_account()
    # print(gc)

    # Create a connection object.
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
        ],
    )
    conn = connect(credentials=credentials)


    sheet_url = st.secrets["sheet_url"]
    rows = run_query(f'SELECT * FROM "{sheet_url}"')

    # Print results.
    for row in rows:
        st.write(f"{row.name} has a :{row.pet}:")

    # Obtain link to google sheet
    sheets_url = st.secrets["sheet_url"]

    # RD 1
    number = 1
    subvariables = 4
    score_1 = riskDriver(subvariables, number)

    # RD 1
    number = 2
    subvariables = 3
    score_2 = riskDriver(subvariables, number)


    total_score = (score_1 + score_2)/2

    st.text('Final Score: ' + str(total_score))

    send = st.button("Submit")

    # TODO save as dataframe/csv
    # dataframe = pd.DataFrame()

    # if send:
    #     write_data(sheets_url) 

    # if send:
    #     with open('testfile.txt', 'w') as f:
    #         f.write('Final Score: ' + str(total_score))


    
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