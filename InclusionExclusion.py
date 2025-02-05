import streamlit as st
import pandas as pd
import requests
import json
import time
from urllib.parse import quote
from langchain.chat_models import ChatOpenAI

def validate_nct_id(nct_id):
    """Validate NCT ID format according to ClinicalTrials.gov specifications"""
    return (
        isinstance(nct_id, str) and 
        nct_id.startswith('NCT') and 
        len(nct_id) == 11 and 
        nct_id[3:].isdigit()
    )

def fetch_trial_criteria(nct_id):
    """Retrieve eligibility criteria from ClinicalTrials.gov API v2"""
    api_url = f"https://clinicaltrials.gov/api/v2/studies/{quote(nct_id)}"
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

def parse_criteria(llm, criteria_text):
    """Parse criteria text using LLM with enhanced validation"""
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

def correlate_patients_with_trials(patient_df, trial_df):
    correlated_results = []
    
    for _, patient_row in patient_df.iterrows():
        for _, trial_row in trial_df.iterrows():
            if trial_row['Type'] == 'Inclusion':
                # Check if any word from the patient's diagnosis is in the inclusion criteria
                patient_diagnosis_words = patient_row['Primary Diagnosis'].lower().split()
                criteria_words = trial_row['Criteria'].lower().split()
                matched_criteria = [word for word in patient_diagnosis_words if word in criteria_words]
                
                if matched_criteria:
                    correlated_results.append({
                        'Patient Name': patient_row['Patient Name'],
                        'Patient ID': patient_row['Patient ID'],
                        'NCT Number': trial_row['NCT Number'],
                        'Study Title': trial_row['Study Title'],
                        'Type': 'Eligible',
                        'Matched Criteria': ', '.join(matched_criteria),
                        'Criteria Type': 'Inclusion',
                        'Full Criteria': trial_row['Criteria']
                    })
            elif trial_row['Type'] == 'Exclusion':
                # Check if any word from the patient's diagnosis is in the exclusion criteria
                patient_diagnosis_words = patient_row['Primary Diagnosis'].lower().split()
                criteria_words = trial_row['Criteria'].lower().split()
                matched_criteria = [word for word in patient_diagnosis_words if word in criteria_words]
                
                if matched_criteria:
                    correlated_results.append({
                        'Patient Name': patient_row['Patient Name'],
                        'Patient ID': patient_row['Patient ID'],
                        'NCT Number': trial_row['NCT Number'],
                        'Study Title': trial_row['Study Title'],
                        'Type': 'Not Eligible',
                        'Matched Criteria': ', '.join(matched_criteria),
                        'Criteria Type': 'Exclusion',
                        'Full Criteria': trial_row['Criteria']
                    })
    
    return pd.DataFrame(correlated_results)

st.title("Clinical Trial Criteria Batch Processor")
openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')

uploaded_file = st.file_uploader("Upload Clinical Trial Criteria File", type=["xlsx", "xls", "csv"],
                                help="Supports Excel & CSV files with 'NCT Number' and 'Study Title' columns")

patient_database_file = st.file_uploader("Upload Patient Database", type=["xlsx", "xls", "csv"],
                                        help="Supports Excel & CSV files with patient data")

if uploaded_file:
    try:
        if uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file, engine='openpyxl', dtype=str)
        else:
            df = pd.read_csv(uploaded_file, 
                           encoding='utf-8-sig',
                           engine='python',
                           dtype=str)

        required_cols = {'NCT Number', 'Study Title'}
        if not required_cols.issubset(df.columns):
            st.error("Uploaded file must contain 'NCT Number' and 'Study Title' columns")
            st.stop()

        df['NCT Number'] = df['NCT Number'].str.strip().str.upper()
        invalid_ids = df[~df['NCT Number'].apply(validate_nct_id)]['NCT Number'].tolist()
        
        if invalid_ids:
            st.error(f"Invalid NCT ID format detected: {', '.join(invalid_ids[:3])}...")
            st.stop()

        llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-4", temperature=0.1)
        results = {}
        
        # Initialize progress bar
        progress_bar = st.progress(0)
        total_trials = len(df)
        
        for index, row in df.iterrows():
            nct_id = row['NCT Number']
            st.write(f"Processing {nct_id}...")
            
            criteria_text = fetch_trial_criteria(nct_id)
            time.sleep(1)  # Rate limiting
            
            if not criteria_text:
                continue
            
            parsed = parse_criteria(llm, criteria_text)
            
            # Group results by NCT ID
            results[nct_id] = {
                'Study Title': row['Study Title'],
                'Inclusion Criteria': parsed.get('inclusion', []),
                'Exclusion Criteria': parsed.get('exclusion', [])
            }
            
            # Update progress bar
            progress = (index + 1) / total_trials
            progress_bar.progress(progress)
        
        progress_bar.empty()  # Clear progress bar after completion
        
        if results:
            # Convert grouped results to DataFrame
            output_data = []
            for nct_id, result in results.items():
                output_data.append({
                    'NCT Number': nct_id,
                    'Study Title': result['Study Title'],
                    'Type': 'Inclusion',
                    'Criteria': '\n'.join(result['Inclusion Criteria'])
                })
                output_data.append({
                    'NCT Number': nct_id,
                    'Study Title': result['Study Title'],
                    'Type': 'Exclusion',
                    'Criteria': '\n'.join(result['Exclusion Criteria'])
                })
            
            output_df = pd.DataFrame(output_data)
            st.subheader("Preview")
            st.dataframe(output_df.head(10))
            
            csv = output_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download CSV Results",
                data=csv,
                file_name="clinical_trial_criteria.csv",
                mime="text/csv"
            )
        else:
            st.warning("No valid criteria found in uploaded trials")
                
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")

if patient_database_file:
    try:
        if patient_database_file.name.endswith(('.xlsx', '.xls')):
            patient_df = pd.read_excel(patient_database_file, engine='openpyxl', dtype=str)
        else:
            patient_df = pd.read_csv(patient_database_file, 
                                   encoding='utf-8-sig',
                                   engine='python',
                                   dtype=str)
        
        if 'output_df' in locals():
            correlated_df = correlate_patients_with_trials(patient_df, output_df)
            
            st.subheader("Correlated Results")
            st.dataframe(correlated_df.head(10))
            
            csv = correlated_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Correlated Results",
                data=csv,
                file_name="patient_trial_correlation.csv",
                mime="text/csv"
            )
        else:
            st.warning("Please upload clinical trial criteria first.")
            
    except Exception as e:
        st.error(f"Error processing patient database: {str(e)}")
