import streamlit as st
import pandas as pd
import requests
import json
from langchain.chat_models import ChatOpenAI

# Function to validate NCT ID format
def validate_nct_id(nct_id):
    return (
        isinstance(nct_id, str) and 
        nct_id.startswith('NCT') and 
        len(nct_id) == 11 and 
        nct_id[3:].isdigit()
    )

# Function to fetch trial criteria from ClinicalTrials.gov API
def fetch_trial_criteria(nct_id):
    api_url = f"https://clinicaltrials.gov/api/v2/studies/{nct_id}"
    params = {
        "format": "json",
        "markupFormat": "markdown"
    }
    headers = {
        "Accept": "text/csv, application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    try:
        response = requests.get(api_url, headers=headers, params=params, timeout=15)
        
        if response.status_code == 400:
            st.error(f"Invalid request for {nct_id} - check API parameters")
            return None
            
        if response.status_code != 200:
            st.error(f"API Error {response.status_code} for {nct_id}")
            return None

        data = response.json()
        
        # Validate API response structure
        eligibility_module = data.get('protocolSection', {}).get('eligibilityModule', {})
        if not eligibility_module:
            st.error(f"No eligibility data found for {nct_id}")
            return None
            
        return eligibility_module.get('eligibilityCriteria')

    except Exception as e:
        st.error(f"Error fetching {nct_id}: {str(e)}")
        return None

# Function to parse criteria text using LLM
def parse_criteria(llm, criteria_text):
    if not criteria_text or len(criteria_text.strip()) < 50:
        return {"inclusion": [], "exclusion": []}
    
    prompt = f"""Convert this clinical trial criteria into JSON format with separate inclusion/exclusion lists.
    Use exactly this structure:
    {{
        "inclusion": ["list", "of", "criteria"],
        "exclusion": ["list", "of", "criteria"]
    }}
    
    Input Text:
    {criteria_text}
    """
    try:
        result = llm.invoke(prompt).content
        parsed = json.loads(result.strip('` \n'))
        
        # Validate response structure
        if not isinstance(parsed.get('inclusion'), list) or not isinstance(parsed.get('exclusion'), list):
            raise ValueError("Invalid LLM response structure")
            
        return parsed
    except json.JSONDecodeError:
        st.error("Failed to parse LLM response as JSON")
        return {"inclusion": [], "exclusion": []}
    except Exception as e:
        st.error(f"Parsing error: {str(e)}")
        return {"inclusion": [], "exclusion": []}

# Function to correlate patients with trials
def correlate_patient_with_trial(patient_row, trial_row):
    patient_diagnosis_words = patient_row['Primary Diagnosis'].lower().split()
    criteria_words = trial_row['Criteria'].lower().split()
    
    if any(word in criteria_words for word in patient_diagnosis_words):
        return "Yes"
    else:
        return "No"

# Sidebar for OpenAI API Key
with st.sidebar:
    openai_api_key = st.text_input("Enter OpenAI API Key")

# Load files
uploaded_files = st.file_uploader("Upload files", type=["xlsx", "xls", "csv"], accept_multiple_files=True)

if len(uploaded_files) >= 2 and openai_api_key:
    clinical_trial_file = uploaded_files[0]
    patient_database_file = uploaded_files[1]
    
    # Load data
    trial_df = pd.read_excel(clinical_trial_file, engine='openpyxl', dtype=str)
    patient_df = pd.read_excel(patient_database_file, engine='openpyxl', dtype=str)
    
    # Extract NCT numbers and patient names
    nct_numbers = trial_df['NCT Number'].tolist()
    patient_names = patient_df['Patient Name'].tolist()
    
    # Create dropdowns
    selected_nct = st.selectbox("Select NCT Number", nct_numbers)
    selected_patient = st.selectbox("Select Patient Name", patient_names)
    
    # Initialize LLM
    llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-4", temperature=0.1)
    
    # Button to check eligibility
    if st.button("Check Eligibility"):
        st.write("Checking eligibility...")
        
        # Fetch and parse criteria for selected trial
        criteria_text = fetch_trial_criteria(selected_nct)
        if criteria_text:
            parsed_criteria = parse_criteria(llm, criteria_text)
            
            # Display inclusion criteria in a table
            inclusion_table = []
            for i, criterion in enumerate(parsed_criteria['inclusion'], start=1):
                eligibility = "Yes" if criterion.lower() in patient_df[patient_df['Patient Name'] == selected_patient].iloc[0]['Primary Diagnosis'].lower() else "No"
                inclusion_table.append({
                    'Criterion #': i,
                    'Inclusion Criterion': criterion,
                    'Is Patient Included': eligibility
                })
            
            inclusion_df = pd.DataFrame(inclusion_table)
            st.write("### Inclusion Criteria:")
            st.dataframe(inclusion_df)
            
            # Display exclusion criteria in a table
            exclusion_table = []
            for i, criterion in enumerate(parsed_criteria['exclusion'], start=1):
                eligibility = "Yes" if criterion.lower() in patient_df[patient_df['Patient Name'] == selected_patient].iloc[0]['Primary Diagnosis'].lower() else "No"
                exclusion_table.append({
                    'Criterion #': i,
                    'Exclusion Criterion': criterion,
                    'Is Patient Excluded': eligibility
                })
            
            exclusion_df = pd.DataFrame(exclusion_table)
            st.write("### Exclusion Criteria:")
            st.dataframe(exclusion_df)
        else:
            st.error("No eligibility criteria found for the selected trial.")
