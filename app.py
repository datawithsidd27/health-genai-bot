import streamlit as st
import google.generativeai as genai
import pandas as pd
import sqlite3
import os
import time

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
st.set_page_config(page_title="GenAI Health Analyst", layout="wide", page_icon="🧬")

def get_api_key():
    if "GEMINI_API_KEY" in st.secrets:
        return st.secrets["GEMINI_API_KEY"]
    return os.getenv("GEMINI_API_KEY")

api_key = get_api_key()
if not api_key:
    st.error("⚠️ API Key missing! Please set 'GEMINI_API_KEY' in Streamlit Secrets.")
    st.stop()

genai.configure(api_key=api_key)

# Use the safe alias for the Free Tier model
MODEL_NAME = 'models/gemini-flash-latest'

# ==========================================
# 2. ROBUST AI CALLER
# ==========================================
def ask_gemini(prompt, retries=3):
    model = genai.GenerativeModel(MODEL_NAME)
    for attempt in range(retries):
        try:
            return model.generate_content(prompt).text
        except Exception as e:
            if attempt == retries - 1:
                st.error(f"AI Error: {e}")
                return None
            time.sleep(2)
    return None

# ==========================================
# 3. DATABASE SETUP (EXACT SCHEMA MATCH)
# ==========================================
@st.cache_resource
def setup_database():
    if not os.path.exists("genetics.csv") or not os.path.exists("lifestyle.csv"):
        st.error("❌ CSV files missing in GitHub repo!")
        st.stop()

    genetics_df = pd.read_csv("genetics.csv")
    lifestyle_df = pd.read_csv("lifestyle.csv")
    
    conn = sqlite3.connect(':memory:', check_same_thread=False)
    genetics_df.to_sql('genetics', conn, index=False)
    lifestyle_df.to_sql('lifestyle', conn, index=False)
    return conn

conn = setup_database()

# ==========================================
# 4. AGENT LOGIC (FIXED COLUMN NAMES)
# ==========================================
def get_sql_query(user_question):
    # CRITICAL: We list the EXACT column names here to prevent "no such column" errors
    prompt = f"""
    You are an expert SQLite Data Analyst. Convert the user's question into a valid SQL query.
    
    ### Database Schema (Exact Column Names):
    
    1. **Table: genetics**
       - Patient_Number (INTEGER)
       - Blood_Pressure_Abnormality (0 or 1)
       - Level_of_Hemoglobin (FLOAT)
       - Genetic_Pedigree_Coefficient (FLOAT)
       - Age (INTEGER)
       - BMI (INTEGER)
       - Sex (0 or 1)
       - Pregnancy (0 or 1)
       - Smoking (0 or 1)  <-- NOTE: Column name is 'Smoking', NOT 'Smoker'
       - salt_content_in_the_diet (INTEGER)
       - alcohol_consumption_per_day (FLOAT)
       - Level_of_Stress (1, 2, 3)
       - Chronic_kidney_disease (0 or 1)
       - Adrenal_and_thyroid_disorders (0 or 1)
       
    2. **Table: lifestyle**
       - Patient_Number (INTEGER)
       - Day_Number (INTEGER)
       - Physical_activity (FLOAT)
       
    ### Rules:
    - Join tables on `Patient_Number`.
    - If asking about 'Physical Activity', use `AVG(Physical_activity)` as there are multiple rows per patient.
    - Return ONLY the raw SQL query. Do NOT use markdown blocks (```sql).
    
    ### Question:
    "{user_question}"
    """
    
    response = ask_gemini(prompt)
    if response:
        # Clean up any potential markdown formatting
        return response.strip().replace('```sql', '').replace('```', '')
    return None

def generate_natural_insight(question, df):
    data_str = df.to_markdown()
    prompt = f"Question: {question}\nData:\n{data_str}\nProvide a concise health insight."
    return ask_gemini(prompt)

# ==========================================
# 5. FRONTEND
# ==========================================
st.title("🧬 GenAI Health Chatbot")
st.caption("Powered by Gemini 1.5 Flash")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "chart_data" in msg:
            st.bar_chart(msg["chart_data"])

if prompt := st.chat_input("Ask: 'Compare BMI of smokers vs non-smokers'"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    with st.spinner("Analyzing..."):
        try:
            sql = get_sql_query(prompt)
            if sql:
                df = pd.read_sql_query(sql, conn)
                
                if df.empty:
                    response = "No matching records found."
                    chart_data = None
                else:
                    response = generate_natural_insight(prompt, df)
                    
                    # Auto-Chart Logic
                    chart_data = None
                    numeric_cols = df.select_dtypes(include=['number']).columns
                    if len(numeric_cols) > 0 and len(df) > 1:
                        non_numeric = df.select_dtypes(exclude=['number']).columns
                        chart_df = df.set_index(non_numeric[0]) if len(non_numeric) > 0 else df
                        chart_data = chart_df[numeric_cols]

                with st.chat_message("assistant"):
                    st.markdown(response)
                    if chart_data is not None:
                        st.bar_chart(chart_data)
                    with st.expander("Debug SQL"):
                        st.code(sql)
                        st.dataframe(df)

                msg_payload = {"role": "assistant", "content": response}
                if chart_data is not None: msg_payload["chart_data"] = chart_data
                st.session_state.messages.append(msg_payload)

        except Exception as e:
            st.error(f"Error: {e}")
