import streamlit as st
import pandas as pd
import requests
import json
from langchain.chat_models import ChatOpenAI
import os
import time

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
def correlate_patient_with_trial(llm, patient_row, criterion):
    """Use LLM to determine if a patient meets a specific criterion."""
    
    # Prepare patient information
    patient_info = {
        'primary_diagnosis': patient_row['Primary Diagnosis'],
        'secondary_diagnosis': patient_row['Secondary Diagnosis'],
        'prescription': patient_row['Prescription'],
        'jcode': patient_row['JCode']
    }
    
    # Construct prompt for LLM
    prompt = f"""
    Determine if the patient meets the following criterion based on their medical information.Answer with "Yes" or "No" only.

    **Patient Information:**
    - Primary Diagnosis: {patient_info['primary_diagnosis']}
    - Secondary Diagnosis: {patient_info['secondary_diagnosis']}
    - Prescription: {patient_info['prescription']}
    - JCode: {patient_info['jcode']}

    **Criterion:**
    {criterion}

    Is the patient eligible?
    """
    
    try:
        result = llm.invoke(prompt).content
        eligibility = result.strip('` \n')
        if eligibility.lower() not in ['yes', 'no']:
            return "No"  # Default to "No" if response is not clear
            
        return eligibility.capitalize()
    except Exception as e:
        st.error(f"Error determining eligibility: {str(e)}")
        return "Unknown"

#Main Funciton
st.set_page_config(page_title="Patient Trial Eligibility Checker", page_icon="🩺", layout="wide")
#st.image("Mool.png", width=100)

col1, col2 = st.columns([1, 6])
with col1:
    st.image("Mool.png", width=150)

with col2:
    st.markdown(
        "<h1 style='margin-top: 10px;'>Patient Trial Eligibility Checker</h1>",
        unsafe_allow_html=True
    )

# Sidebar for OpenAI API Key
#with st.sidebar:
#    openai_api_key = st.text_input("Enter OpenAI API Key")
with st.spinner("🔄 Mool AI agent Authentication In progress..."):
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        st.error("❌ API_KEY not found in environment variables.")
        st.stop()
    time.sleep(5)
st.success("✅ Mool AI agent Authentication Successful")


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
            selected_patient_row = patient_df[patient_df['Patient Name'] == selected_patient].iloc[0]
            inclusion_score_numerator=0
            criterion_number = 1
            for i, criterion in enumerate(parsed_criteria['inclusion'], start=1):
                if criterion.strip().lower().startswith("registration #"):
                    continue
                eligibility = correlate_patient_with_trial(llm, selected_patient_row, criterion)
                inclusion_table.append({
                    'Criterion': criterion_number,
                    'Inclusion Criterion': criterion,
                    'Is Patient Included': eligibility
                })
                criterion_number += 1
                if eligibility == "Yes":
                    inclusion_score_numerator += 1
            
            inclusion_df = pd.DataFrame(inclusion_table)
            st.write("### Inclusion Criteria:")
            st.dataframe(inclusion_df)
            

            inclusion_score_denominator = len(inclusion_table)
            if inclusion_score_denominator > 0:
                inclusion_score = (inclusion_score_numerator / inclusion_score_denominator) * 100
                st.write(f"**Inclusion Score:** {inclusion_score_numerator} / {inclusion_score_denominator} = {inclusion_score:.2f}%")
            else:
                st.write("No inclusion criteria to evaluate.")
            
            # Display exclusion criteria in a table
            exclusion_table = []
            exclusion_score_numerator=0
            criterion_number = 1
            for i, criterion in enumerate(parsed_criteria['exclusion'], start=1):
                eligibility = correlate_patient_with_trial(llm, selected_patient_row, criterion)
                exclusion_table.append({
                    'Criterion': criterion_number,
                    'Exclusion Criterion': criterion,
                    'Is Patient Excluded': eligibility
                })
                criterion_number += 1
                if eligibility == "Yes":
                    exclusion_score_numerator += 1
            
            exclusion_df = pd.DataFrame(exclusion_table)
            st.write("### Exclusion Criteria:")
            st.dataframe(exclusion_df)
            

            
            exclusion_score_denominator = len(exclusion_table)
            if exclusion_score_denominator > 0:
                exclusion_score = (exclusion_score_numerator / exclusion_score_denominator) * 100
                st.write(f"**Exclusion Score:** {exclusion_score_numerator} / {exclusion_score_denominator} = {exclusion_score:.2f}%")
            else:
                st.write("No exclusion criteria to evaluate.")
        else:
            st.error("No eligibility criteria found for the selected trial.")
