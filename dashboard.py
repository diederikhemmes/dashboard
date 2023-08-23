# -----------------------------------------------------------------------------
# Title: Circularity Score Dashboard
# Author: Iris Reitsma
# Date: August 23, 2023
# Description: This script creates a streamlit dashboard. This dashboard calculates 
#   a circularity score given input parameters corresponding to a circular economy 
#   company. A user can select which expert input to incorporate in the calculation,
#   and can choose whether to score risk drivers using multiple variables or one
#   overwrite. 
# -----------------------------------------------------------------------------

import streamlit as st
import pandas as pd
import datetime
import pytz 
from PIL import Image
import gspread
from google.oauth2 import service_account
import matplotlib.pyplot as plt
import numpy as np

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
    st.image(logos)

    return


@st.cache_data
def load_expert_data(filepath_weights: str) -> pd.DataFrame:
    """
    Load expert data from an Excel file.

    This function reads an Excel file containing expert weights data and returns
    the data as a pandas DataFrame.

    Args:
        filepath_weights (str): File path to the Excel file containing expert weights data.

    Returns:
        pd.DataFrame: DataFrame containing the loaded expert weights data.
    """
    weights_df = pd.read_excel(filepath_weights)

    return weights_df


@st.cache_data
def load_drivers_data(filepath_risk: str) -> pd.DataFrame:
    """
    Load risk drivers data from an Excel file.

    This function reads an Excel file containing risk drivers data and returns
    the data as a pandas DataFrame. 

    Args:
        filepath_risk (str): File path to the Excel file containing risk drivers data.

    Returns:
        pd.DataFrame: DataFrame containing the loaded risk drivers data with missing
        values filled using forward fill.
    """
    drivers_df = pd.read_excel(filepath_risk)
    drivers_df = drivers_df.fillna(method='ffill', axis=0)

    return drivers_df


@st.cache_data
def load_peak_data(filepath_peaks: str) -> pd.DataFrame:
    """
    Load peak extraction data from an Excel file.

    This function reads an Excel file containing peak extraction data and returns
    the data as a pandas DataFrame.

    Args:
        filepath_weights (str): File path to the Excel file containing peak extraction data.

    Returns:
        pd.DataFrame: DataFrame containing the loaded peak extraction data.
    """
    peaks_df = pd.read_excel(filepath_peaks)
    peaks_df = peaks_df.fillna('no examples')

    return peaks_df

    
def riskDriver(riskdriver_groupby: pd.DataFrame, riskdriver: str, number: int) \
    -> tuple[float, str, list[float], list[str], float, str]:
    """
    Calculate risk driver scores and provide user interface for customization.

    This function calculates risk driver scores based on user inputs for various
    variables. It presents an interactive user interface for selecting options
    and overwriting scores for each risk driver.

    Args:
        riskdriver_groupby (pd.DataFrame): Grouped data containing risk driver information.
        riskdriver (str): Name of the risk driver.
        number (int): Sequential number of the risk driver.

    Returns:
        tuple[float, str, list[float], list[str], float, str]: A tuple containing:
            - The calculated normalized score for the risk driver.
            - The score in text.
            - A list of normalized scores for variables.
            - A list of selected text score per variable.
            - The selected overwrite score.
            - The selected overwrite option in text.
    """

    # Define column width
    col1, col2 = st.columns([10, 5])

    # Retrieve driver information from groupby object
    variables = riskdriver_groupby['Variable'].unique().tolist()
    overwrites_init = riskdriver_groupby['Overwrites'].unique().tolist()
    overwrites = ['Overwrite: '+ i for i in overwrites_init]

    # Create drop down for riskdriver
    with col1:
        with st.expander('Risk Driver ' + str(number) + ': ' + riskdriver):
            inputs = [None]*len(variables)
            inputs_text = [None]*len(variables)

            # Loop over variables'
            min_score_rel = 0
            max_score = 0
            scored_points = 0
            for i, v in enumerate(variables):
                options = riskdriver_groupby[riskdriver_groupby['Variable']==v]['Answers'].tolist()
                examples = riskdriver_groupby[riskdriver_groupby['Variable']==v]['Example'].tolist()
                string = generate_help(options, examples)
                choice = st.selectbox(v, options, help=string)

                # Determine score for variable (v)
                v_score = options.index(choice) + 1
                v_score_rel = v_score / len(options)
                inputs[i] = v_score_rel
                inputs_text[i] = choice
                scored_points += v_score
                max_score += len(options)
                min_score_rel += 1 / len(options)

            # Determince score for risk driver
            score = sum(inputs)
            score_rel = score / len(variables)
            min_score_rel = min_score_rel / len(variables)
            score_rel_text = overwrites_init[round(scored_points/max_score*(len(overwrites_init)))-1]
            overwrites.insert(0, score_rel_text)

    # Create input box for overwriting risk driver score
    with col2:
        overwrite = False
        overwrite_text = st.selectbox('Overwrite', overwrites, label_visibility='collapsed')
        if overwrite_text != score_rel_text:
            score = overwrites.index(overwrite_text) 
            score_rel_overwrite = score / (len(overwrites_init))
            score_rel = convert_range(score_rel_overwrite, 1/(len(overwrites_init)), 1, min_score_rel, 1) # Scale overwrite to same range as based variables
            score_rel_text = overwrite_text
            overwrite = True

    return score_rel, score_rel_text, inputs, inputs_text, overwrite, overwrite_text

@st.cache_data
def generate_help(options: list[str], examples: list[str]) -> str:
    """
    Generate a help string with examples for each option.

    This function generates a formatted help string containing examples for each
    option provided. The options and their corresponding examples are combined
    into a user-friendly format.

    Args:
        options (list[str]): A list of options for a variable.
        examples (list[str]): A list of examples corresponding to each option.

    Returns: 
        str: A formatted help string containing option names and examples.
    """ 
    help_string = 'These are examples of the answers: \n\n' 
    for n, t in enumerate(options):
        help_string += t
        help_string += ': '
        help_string += """ \n """
        help_string += examples[n]
        help_string += '\n\n' 
    
    return help_string

@st.cache_data
def convert_range(value: float, min_original: float, max_original: float, \
                  min_new: float, max_new:float) -> float:
    """
    Convert a value from one range to another range.

    This function takes a value within a specified original range and scales
    it to a new range defined by the given minimum and maximum values.

    Args:
        value (float): The value to be converted.
        min_original (float): The minimum value of the original range.
        max_original (float): The maximum value of the original range.
        min_new (float): The minimum value of the new range.
        max_new (float): The maximum value of the new range.

    Returns:
        float: The scaled value within the new range.
    """
    # Calculate the range of the original values
    range_original = max_original - min_original

    # Scale and shift the value to the new range
    scaled_value = (value - min_original) / range_original * (max_new - min_new) + min_new

    return scaled_value


@st.cache_data
def prepare_weights(weights_df: pd.DataFrame, finance_weight: bool, risk_weight: bool, 
                    invest_weight: bool, busdev_weight: bool) -> pd.DataFrame:
    """
    Prepare weighted scores based on selected expert groups.

    This function prepares the weights for different riskdrivers based on 
    user-selected expert groups. It calculates the weighted average of the selected 
    expert groups and returns the resulting weights for further analysis.

    Args:
        weights_df (pd.DataFrame): DataFrame containing expert groups and their scores.
        finance_weight (bool): If True, include Finance expert group in calculations.
        risk_weight (bool): If True, include Risk expert group in calculations.
        invest_weight (bool): If True, include Invest expert group in calculations.
        busdev_weight (bool): If True, include Business Development expert group in calculations.

    Returns:
        pd.DataFrame: A DataFrame containing calculated weights for each risk driver.
    """
   
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

    # Calculate the weighted average of the selected weights
    selected_weights_df = weights_df[selected_groups]
    weights_df['Sum'] = selected_weights_df.sum(axis=1)
    total_points = weights_df['Sum'].sum()
    weights_df['Weight'] = weights_df['Sum'] / total_points

    # st.dataframe(weights_df)

    return weights_df 


def create_weights_UI() -> tuple[bool, bool, bool, bool]:
    """
    Create a user interface for choosing expert weights.

    This function generates a user interface with checkboxes for selecting expert
    weights from different expert groups. It allows users to choose which expert
    groups to include in the calculations.

    Returns:
        tuple[bool, bool, bool, bool]: A tuple containing Boolean values indicating whether
        each of the expert weights (Finance, Risk, Invest, Bus Dev) is selected.

    """
    st.header('Choose expert weights')

    # Create checkboxes for each expert group
    col1, col2, col3, col4 = st.columns([1,1,1,1.3])
    with col1:
        finance_weight = st.checkbox('Finance', value=True)
    with col2:
        risk_weight = st.checkbox('Risk', value=True)
    with col3:
        invest_weight = st.checkbox('Invest', value=True)
    with col4:
        busdev_weight = st.checkbox('Business Development', value=True)

    # At least one of the weights should be selected, display error if not
    if not any([finance_weight, risk_weight, invest_weight, busdev_weight]):
        st.error('Please select at least one expert weight', icon="🚨")

    return finance_weight, risk_weight, invest_weight, busdev_weight


def apply_weights(weights_df: pd.DataFrame, risk_driver: str, value: float) -> float:
    """
    Apply a weight to a given value based on a specific risk driver.

    This function applies a weight from a DataFrame of weights to a provided risk 
    driver score. It calculates the weighted value using the weight associated 
    with the given risk driver.

    Args:
        weights_df (pd.DataFrame): DataFrame containing risk driver weights.
        risk_driver (str): Name of the risk driver for which the weight should be applied.
        value (float): The value to be weighted.

    Returns:
        float: The value after applying the specified weight.
    """
    # Obtain and apply weight
    weight = weights_df.loc[weights_df['Risk Factor'] == risk_driver, 'Weight'].values[0]
    weighted_value = value * weight

    return weighted_value

@st.cache_data
def plot_weights(weights_df_plot: pd.DataFrame) -> None:
    """
    Create a bar chart depicting the distribution of importance points over risk drivers,
    colored by expert group.

    Parameters:
        weights_df_plot (pandas.DataFrame): DataFrame containing importance weights per risk driver.

    Returns:
        None
    """
    plt.style.use('tableau-colorblind10')   

    # Normalize expert points
    column_sums = weights_df_plot.sum()
    normalized_df = weights_df_plot.iloc[:, 1:].div(column_sums, axis=1)
    normalized_df['Risk Factor'] = weights_df_plot['Risk Factor']

    # Set the 'Risk Factor' column as the index
    normalized_df.set_index('Risk Factor', inplace=True)

    # Get the column names and row indices
    columns = normalized_df.columns
    indices = normalized_df.index
    width = 0.1

    # Create an array for x-axis positions
    x = np.arange(len(indices))

    # Plotting the bar chart 
    fig, ax = plt.subplots(figsize=(10,5))
    for i, column in enumerate(columns):
        ax.bar(x + i * width, normalized_df[column], width=width, label=column) 

    # Adding labels, legend, and title 
    ax.set_xticks(x + (len(columns) - 1) * width / 2)
    ax.set_xticklabels(indices, rotation=45, ha='right', fontsize=12)
    ax.set_ylabel('Fraction of points', fontsize=14)
    ax.legend()

    st.pyplot(fig)

    return 


#### main ##################################################################################################
if __name__ == '__main__':

    # Create logo header
    image_path = 'Images/Logos1.png'
    load_images(image_path)

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Scorecard", "Read before use", "Peak extraction years", "Distribution of expert weights"])

    # Set up the scope and credentials to Google Sheet
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
        ],
    )

    # Obtain general information
    date = datetime.datetime.now(tz=pytz.timezone('Europe/Amsterdam')).date()
    time = datetime.datetime.now(tz=pytz.timezone('Europe/Amsterdam')).time().replace(microsecond=0)
    version = 'v0.61'

#####################################################################################  
# Scorecard tab
#####################################################################################  
    with tab1:

        # Try to open sheet
        try:
            client = gspread.authorize(credentials)
            sheets_url = st.secrets["sheet_url"]
            sh = client.open_by_url(sheets_url)
        except:
            st.error('Please check your internet connection. If this is not the issue, the connection to the sheet is lost.', icon="🚨")

        # Create name and company input fields
        st.header('Fill in details')
        user = st.text_input(label='Filled in by', placeholder='Your name/company', key='name_key')
        client_company = st.text_input(label='Filled in for', placeholder='Name/company for which to determine a risk score', key='client_key')

        # Store general information
        row = [str(date), str(time), version, user, client_company]
        text_row = [] # Row for storing variable selections in text
        info_columns = len(row)

        # Create expert weight selection UI
        finance_weight, risk_weight, invest_weight, busdev_weight = create_weights_UI()

        # Read in expert weights file 
        filepath_weights = 'Data/Expert_weights.xlsx'
        weights_df_init = load_expert_data(filepath_weights)

        # Prepare weights based on selected expert groups
        weights_df = prepare_weights(weights_df_init, finance_weight, risk_weight, invest_weight, busdev_weight)

        st.header('Score risk factors')

        # Read in risk driver file
        filepath_risk = 'Data/Risk_drivers.xlsx'
        drivers_df = load_drivers_data(filepath_risk)

        # Create dropdown for each risk driver
        n = 1
        risk_scores = []
        for name, rd in drivers_df.groupby('Risk Driver', sort=False):
            risk, risk_text, inputs, inputs_text, overwrite, overwrite_text = riskDriver(rd, name, n)
            
            # Apply expert weight
            weighted_risk = apply_weights(weights_df, name, risk)

            # Store risk driver information
            risk_scores.append(weighted_risk)
            row.extend(inputs)
            row.append(risk)
            row.append(overwrite)
            text_row.extend(inputs_text)
            text_row.append(risk_text)
            text_row.append(overwrite_text)
            n += 1

        st.header('Determine final score')
 
        # Determine final score and convert in on scale 0 to 100
        final_score = sum(risk_scores)
        min_score = 0.266246 # Lowest score possible when filling in form
        normalized_final_score = convert_range(final_score, min_original=min_score, max_original=1, min_new=0, max_new=100)

        # Display final score and insert to correct column
        col1, col2 = st.columns([3,2])
        with col1:
            st.metric('Circularity Score: ', str(round(normalized_final_score, 1)) + ' / 100')
            st.progress(round(normalized_final_score, 1)/100)

        with col2:
            internal_score = st.number_input('Internal PD (0.0100 = 1%)', min_value=0.000, max_value=1.000, step=0.0001, format="%.4f")
        
        # Store scorecard score and internal PD
        row.insert(info_columns, normalized_final_score)
        row.insert(info_columns+1, internal_score)
        row.extend([finance_weight, risk_weight, invest_weight, busdev_weight])

        st.header('Feedback and submit')

        # Request feedback
        feedback = st.text_area('Feedback on this form', placeholder="Please provide here any feedback that you have on, e.g., convenience of filing in this form, or how well the score from this form aligns with your internal PD. ", label_visibility='collapsed')
        row.append(feedback)

        # Add variable text input to row
        row.extend(text_row)

        # Create test box
        test = st.checkbox('Please select this box if you are testing the dashboard', value=False)
        row.insert(info_columns, test)

        # Submit data to google sheet
        send = st.button("Submit")
        if send:    
            st.success('Risk score submitted!')
            sh.sheet1.append_row(row)

#####################################################################################  
# Readme tab
#####################################################################################  
    with tab2:
        st.header('Background')
        st.markdown("""This dashboard outputs a circularity score given input parameters corresponding \
                    to a circular economy company. This score can be used as a decision factor in deciding \
                    which circular economy deals are accepted or rejected by, for example, banks. In addition, \
                    the dashboard can give insight into which risk factors are important in making these kinds \
                    of decisions. """)

        st.header('Instructions')
        st.markdown('**Names**: It is possible to use dummy names for the \'filled \
                     in by\' and \'filled in for\' fields.')
        
        st.markdown("""**Scoring risk drivers**: The scorecard contains six risk drivers, \
                    each with multiple variables. Place your mouse on the question mark icon for \
                    examples of how to rate a company. """
        ) 

        st.markdown("""**Overwrite risk score**: If you are unsure about how to rate the \
                    variables, it is possible to provide an overwrite for the risk driver \
                    score. This overwrite replaces the score determined on the separate variables, \
                    which is displayed as the first option in the selection box.""")

        st.markdown("**Expert weights**: Different groups of experts distributed points over \
                    the different risk drivers. It is possible to select expert groups to weight \
                    the circular score. The weight distribution is visualized in the \
                    \'Distribution of expert weights\' tab."
        ) 

        st.markdown("**Circularity score**: The form will show the determined circularity score,\
                     which lies between 0 (lowest) and 100 (good). Please fill in your \
                    internal Probability of Default for comparison.")

#####################################################################################  
# Peak Extraction Year tab
#####################################################################################          
        with tab3:    
            st.markdown("This table contains the expected peak extraction years for materials, which\
                        is needed for risk driver 3.1. Reference: \
                        [Sverdrup & Ragnarsdóttir (2014)](https://www.geochemicalperspectives.org/wp-content/uploads/v3n2.pdf).\
                        ")
            
            # Load and show table
            filepath_peaks = 'Data/Peak_extraction_years.xlsx'    
            peaks_df = load_peak_data(filepath_peaks)
            st.dataframe(peaks_df, hide_index=True, use_container_width=True)

#####################################################################################  
# Expert weight plot tab
#####################################################################################          
        with tab4:
            st.markdown("This bar chart shows what fraction of points was assigned to which risk driver \
                        for each expert group.")
            
            # Plot weights distribution
            plot_weights(weights_df_init[['Risk Factor', 'Sub score Risk', 'Sub score Bus Dev', 'Sub score Finance', 'Sub score Invest']])
            





           

