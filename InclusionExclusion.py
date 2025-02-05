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
        "version": "2.0",  # Changed from 2.0.1 to match API response structure
        "markupFormat": "html"  # Added to match example response format
    }
    headers = {
        "Accept": "application/json",
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
        
        # Validate API response structure based on search results example
        if not data.get('protocolSection', {}).get('eligibilityModule'):
            st.error(f"No eligibility criteria found for {nct_id}")
            return None
            
        return data['protocolSection']['eligibilityModule'].get('eligibilityCriteria')

    except Exception as e:
        st.error(f"Error fetching {nct_id}: {str(e)}")
        return None

def parse_criteria(llm, criteria_text):
    """Parse criteria text using LLM with enhanced validation"""
    if not criteria_text or len(criteria_text.strip()) < 50:
        return {"inclusion": [], "exclusion": []}
    
    # Updated prompt based on example response structure
    prompt = f"""Convert this clinical trial criteria into JSON format with separate inclusion/exclusion lists.
    Use exactly this structure:
    {{
        "inclusion": ["list", "of", "criteria"],
        "exclusion": ["list", "of", "criteria"]
    }}
    
    Input Text (preserve markdown formatting):
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

st.title("Clinical Trial Criteria Batch Processor")
openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')

uploaded_file = st.file_uploader("Upload File", type=["xlsx", "xls", "csv"],
                                help="Supports Excel & CSV files with 'NCT Number' and 'Study Title' columns")

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
        results = []
        
        with st.status("Processing trials...", expanded=True) as status:
            for _, row in df.iterrows():
                nct_id = row['NCT Number']
                st.write(f"Processing {nct_id}...")
                
                criteria_text = fetch_trial_criteria(nct_id)
                time.sleep(1)  # Rate limiting
                
                if not criteria_text:
                    continue
                
                parsed = parse_criteria(llm, criteria_text)
                
                # Add debug information with full criteria text
                debug_info = {
                    'nct_id': nct_id,
                    'criteria_text': criteria_text,
                    'parsed_result': parsed
                }
                st.session_state.setdefault('debug_data', []).append(debug_info)

                for criterion_type in ['inclusion', 'exclusion']:
                    for criterion in parsed.get(criterion_type, []):
                        results.append({
                            'NCT Number': nct_id,
                            'Study Title': row['Study Title'],
                            'Type': criterion_type.capitalize(),
                            'Criterion': criterion
                        })
            
            status.update(label="Processing complete!", state="complete")

        if results:
            output_df = pd.DataFrame(results)
            st.subheader("Preview (First 10 Rows)")
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
            if st.button("Show Debug Info"):
                st.write("Debug Information:")
                st.json(st.session_state.get('debug_data', []))
                
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
