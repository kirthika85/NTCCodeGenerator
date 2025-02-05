import streamlit as st
import pandas as pd
import requests
import json
from langchain.chat_models import ChatOpenAI

def fetch_trial_criteria(nct_id):
    """Retrieve eligibility criteria from ClinicalTrials.gov API"""
    api_url = f"https://clinicaltrials.gov/api/v2/studies/{nct_id}"
    try:
        response = requests.get(api_url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data['studies'][0]['protocolSection']['eligibilityModule']['eligibilityCriteria']
        return None
    except Exception as e:
        st.error(f"Error fetching data for {nct_id}: {str(e)}")
        return None

def parse_criteria(llm, criteria_text):
    """Parse criteria text using LLM with enhanced prompting"""
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
        return json.loads(result.strip('` \n'))
    except Exception as e:
        st.error(f"Parsing error: {str(e)}")
        return {"inclusion": [], "exclusion": []}

st.title("Clinical Trial Criteria Batch Processor")
openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')

uploaded_file = st.file_uploader("Upload File", type=["xlsx", "xls", "csv"],
                                help="Supports Excel & CSV files with 'NCT Code' and 'Study Name' columns")

if uploaded_file:
    try:
        if uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            # Handle CSV with proper encoding
            df = pd.read_csv(uploaded_file, 
                           encoding='utf-8-sig',  # Handles BOM markers
                           engine='python')        # Better error handling
            
        required_cols = {'NCT Code', 'Study Name'}
        
        if not required_cols.issubset(df.columns):
            st.error("Uploaded file must contain 'NCT Code' and 'Study Name' columns")
        else:
            llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-4", temperature=0.1)
            results = []
            
            with st.status("Processing trials...", expanded=True) as status:
                for _, row in df.iterrows():
                    nct_id = row['NCT Code'].strip()
                    st.write(f"Processing {nct_id}...")
                    
                    criteria_text = fetch_trial_criteria(nct_id)
                    if not criteria_text:
                        continue
                    
                    parsed = parse_criteria(llm, criteria_text)
                    
                    # Create structured entries
                    for criterion in parsed.get('inclusion', []):
                        results.append({
                            'NCT Code': nct_id,
                            'Study Name': row['Study Name'],
                            'Type': 'Inclusion',
                            'Criterion': criterion
                        })
                    
                    for criterion in parsed.get('exclusion', []):
                        results.append({
                            'NCT Code': nct_id,
                            'Study Name': row['Study Name'],
                            'Type': 'Exclusion',
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
                
    except UnicodeDecodeError:
        st.error("""Encoding error! Try:
                1. Re-saving CSV with UTF-8 encoding
                2. Try different encoding: pd.read_csv(..., encoding='latin1')""")
