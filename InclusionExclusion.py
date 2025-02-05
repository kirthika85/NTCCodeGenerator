import streamlit as st
import pandas as pd
import requests
import json
from langchain.chat_models import ChatOpenAI

def fetch_trial_criteria(nct_id):
    """Retrieve eligibility criteria from ClinicalTrials.gov API v2"""
    api_url = f"https://clinicaltrials.gov/api/v2/studies/{nct_id}?format=json&version=2.0.1"
    headers = {
        "Accept": "application/json",
        "User-Agent": "ClinicalTrialsProcessor/1.0 (your-contact@email.com)"
    }
    try:
        response = requests.get(api_url, headers=headers, timeout=15)
        if response.status_code != 200:
            st.error(f"API Error {response.status_code} for {nct_id}")
            return None
        
        data = response.json()
        
        # Verify NCT ID matches request
        if data.get('protocolSection', {}).get('identificationModule', {}).get('nctId') != nct_id:
            st.error(f"NCT ID mismatch in response for {nct_id}")
            return None
        
        return data.get('protocolSection', {}).get('eligibilityModule', {}).get('eligibilityCriteria')
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
                                help="Supports Excel & CSV files with 'NCT Number' and 'Study Title' columns")

if uploaded_file:
    try:
        if uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        else:
            df = pd.read_csv(uploaded_file, 
                           encoding='utf-8-sig',
                           engine='python')
        
        required_cols = {'NCT Number', 'Study Title'}
        
        if not required_cols.issubset(df.columns):
            st.error("Uploaded file must contain 'NCT Number' and 'Study Title' columns")
        else:
            llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-4", temperature=0.1)
            results = []
            
            with st.status("Processing trials...", expanded=True) as status:
                for _, row in df.iterrows():
                    nct_id = row['NCT Number'].strip()
                    st.write(f"Processing {nct_id}...")
                    
                    criteria_text = fetch_trial_criteria(nct_id)
                    if not criteria_text:
                        continue
                    
                    parsed = parse_criteria(llm, criteria_text)
                    
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
