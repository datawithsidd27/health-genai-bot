import streamlit as st
import google.generativeai as genai
import pandas as pd
import sqlite3
import os

# ==========================================
# 1. SETUP & AUTHENTICATION
# ==========================================
st.set_page_config(page_title="GenAI Health Analyst", layout="wide", page_icon="🧬")

# Function to get API Key securely (works on both Local & Cloud)
def get_api_key():
    # Try getting from Streamlit Secrets (Best for Cloud)
    if "GEMINI_API_KEY" in st.secrets:
        return st.secrets["GEMINI_API_KEY"]
    # Fallback to Environment Variable (Best for Local/Colab)
    return os.getenv("GEMINI_API_KEY")

api_key = get_api_key()

if not api_key:
    st.error("⚠️ API Key missing! Please set 'GEMINI_API_KEY' in Streamlit Secrets.")
    st.stop()

genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-2.0-flash')

# ==========================================
# 2. DATA LOADING
# ==========================================
@st.cache_resource
def setup_database():
    # Check if files exist
    if not os.path.exists("genetics.csv") or not os.path.exists("lifestyle.csv"):
        st.error("❌ CSV files not found! Please upload 'genetics.csv' and 'lifestyle.csv' to your GitHub repository.")
        st.stop()

    genetics_df = pd.read_csv("genetics.csv")
    lifestyle_df = pd.read_csv("lifestyle.csv")
    
    conn = sqlite3.connect(':memory:', check_same_thread=False)
    genetics_df.to_sql('genetics', conn, index=False)
    lifestyle_df.to_sql('lifestyle', conn, index=False)
    return conn

conn = setup_database()

# ==========================================
# 3. LOGIC: Text-to-SQL Agent
# ==========================================
def get_sql_query(user_question):
    prompt = f"""
    You are an expert Health Data Analyst. Convert this question to a SQL query (SQLite).
    
    Schema:
    1. genetics (Patient_Number, Age, BMI, Blood_Pressure_Abnormality, etc.)
    2. lifestyle (Patient_Number, Day_Number, Physical_activity)
    
    Rules:
    - Join on 'Patient_Number'.
    - If asking about 'Physical Activity', use AVG(Physical_activity) as there are multiple daily entries.
    - Return ONLY raw SQL. No markdown.
    
    Question: "{user_question}"
    """
    response = model.generate_content(prompt)
    return response.text.strip().replace('```sql', '').replace('```', '')

def generate_natural_insight(question, df):
    data_str = df.to_markdown()
    prompt = f"Question: {question}\nData:\n{data_str}\nProvide a concise health insight based on this data."
    return model.generate_content(prompt).text

# ==========================================
# 4. FRONTEND INTERFACE
# ==========================================
st.title("🧬 GenAI Health Chatbot")
st.markdown("Analyze patient data using AI. Ask complex queries joining Genetics and Lifestyle data.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "chart_data" in msg:
            st.bar_chart(msg["chart_data"])

if prompt := st.chat_input("Ask a question (e.g., 'Avg BMI of Smokers?')..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Analyzing..."):
        try:
            sql = get_sql_query(prompt)
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
                with st.expander("View SQL"):
                    st.code(sql)
                    st.dataframe(df)

            msg_payload = {"role": "assistant", "content": response}
            if chart_data is not None: msg_payload["chart_data"] = chart_data
            st.session_state.messages.append(msg_payload)

        except Exception as e:
            st.error(f"Error: {e}")
