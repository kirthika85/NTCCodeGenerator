import streamlit as st
import json
import pandas as pd
from io import StringIO

def parse_eligibility_criteria(uploaded_file):
    try:
        # Read and parse JSON
        data = json.load(uploaded_file)
        
        # Extract NCT ID and criteria
        nct_id = data['studies'][0]['protocolSection']['identificationModule']['nctId']
        criteria = data['studies'][0]['protocolSection']['eligibilityModule']['eligibilityCriteria']

        # Split criteria into inclusion/exclusion sections
        sections = [s.strip() for s in criteria.split("Exclusion Criteria:")]
        
        # Process inclusion criteria
        inclusion = [line.strip() for line in sections[0].split("*") if line.strip()]
        inclusion = [line for line in inclusion if not line.startswith("Inclusion Criteria:")]

        # Process exclusion criteria
        exclusion = [line.strip() for line in sections[1].split("*") if line.strip()]
        
        return nct_id, inclusion, exclusion

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None, [], []

# Streamlit App
st.title("Clinical Trial Criteria Processor")
st.subheader("Upload JSON file to extract inclusion/exclusion criteria")

uploaded_file = st.file_uploader("Choose a JSON file", type="json")

if uploaded_file:
    nct_id, inclusion, exclusion = parse_eligibility_criteria(uploaded_file)
    
    if nct_id:
        st.success(f"Successfully processed: {nct_id}")
        
        # Create DataFrame
        criteria_list = []
        for item in inclusion:
            criteria_list.append({"Type": "Inclusion", "Criterion": item})
        for item in exclusion:
            criteria_list.append({"Type": "Exclusion", "Criterion": item})
            
        df = pd.DataFrame(criteria_list)
        
        # Display results
        st.subheader("Criteria Overview")
        st.dataframe(df)
        
        # Create downloadable CSV
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"{nct_id}_criteria.csv",
            mime="text/csv"
        )
        
        # Show raw criteria
        with st.expander("View Raw Criteria"):
            st.write("### Inclusion Criteria")
            st.write("\n".join(f"- {item}" for item in inclusion))
            
            st.write("### Exclusion Criteria")
            st.write("\n".join(f"- {item}" for item in exclusion))
