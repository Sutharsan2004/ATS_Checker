
import streamlit as st
import PyPDF2
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Resume Analyst", page_icon="🚀", layout="wide")

# --- CSS STYLING (Optional but makes it look pro) ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #FF4B4B;
        color: white;
    }
    .score-card {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR: API KEY & SETTINGS ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=100)
    st.title("Settings")
    
    # Secure API Key Input
    api_key = st.text_input("Enter Groq API Key:", type="password")
    
    
    st.markdown("---")
    st.markdown("### How it works")
    st.markdown("1. Upload your Resume (PDF)")
    st.markdown("2. Paste the Job Description")
    st.markdown("3. Get a Match Score instantly")
    st.markdown("4. Ask AI to fix gaps")

# --- MAIN FUNCTIONS ---

@st.cache_resource
def load_models():
    """Load the heavy models only once to keep the app fast."""
    return SentenceTransformer('all-MiniLm-L6-v2')

def extract_text(uploaded_file):
    """Extract text from uploaded PDF."""
    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + " "
        return text.strip()
    except Exception as e:
        return f"Error: {e}"

def calculate_score(resume_text, jd_text, model):
    """Calculate the match percentage."""
    resume_vec = model.encode([resume_text])
    jd_vec = model.encode([jd_text])
    similarity = cosine_similarity(resume_vec, jd_vec)[0][0]
    return round(similarity * 100, 2)

def generate_ai_feedback(api_key, score, resume_text, jd_text):
    """Call Groq to get improvement suggestions."""
    if not api_key:
        return "⚠️ Please enter your Groq API Key in the sidebar to unlock AI suggestions."
    
    try:
        llm = ChatGroq(temperature=0.3, model_name="llama-3.3-70b-versatile", groq_api_key=api_key)
        
        template = """
        Act as a Senior Technical Recruiter.
        The candidate's resume match score is {score}%.
        
        JOB DESCRIPTION:
        {jd}
        
        RESUME:
        {resume}
        
        Provide a constructive critique in Markdown format:
        1. **Missing Keywords**: List specific technical skills missing.
        2. **Summary Rewrite**: Write a better professional summary.
        3. **Actionable Advice**: 3 bullet points to improve the resume immediately.
        """
        
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | llm
        response = chain.invoke({"score": score, "jd": jd_text, "resume": resume_text})
        return response.content
    except Exception as e:
        return f"Error connecting to AI: {e}"

# --- APP LAYOUT ---

st.title("🚀 Smart Resume Analyzer")
st.subheader("Optimize your resume for ATS with AI")

# Load Model
model = load_models()

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 1. Upload Resume")
    uploaded_file = st.file_uploader("Upload your PDF Resume", type="pdf")

with col2:
    st.markdown("### 2. Job Description")
    jd_text = st.text_area("Paste the Job Description here", height=200)

# --- ACTION BUTTON ---
if st.button("Analyze Resume"):
    if uploaded_file and jd_text:
        with st.spinner("Analyzing..."):
            # 1. Process Text
            resume_text = extract_text(uploaded_file)
            
            # 2. Calculate Score
            score = calculate_score(resume_text, jd_text, model)
            
            # 3. Display Score (Visual Gauge)
            st.markdown("---")
            st.markdown("## 📊 Analysis Results")
            
            # Create a dynamic color for the score
            score_color = "#28a745" if score >= 70 else "#ffc107" if score >= 50 else "#dc3545"
            
            st.markdown(
                f"""
                <div style="background-color: {score_color}; color: white; padding: 20px; border-radius: 10px; text-align: center;">
                    <h1 style="margin:0;">Match Score: {score:.2f}%</h1>
                    <p style="margin:0;">{("Excellent Match" if score >= 70 else "Good Match" if score >= 50 else "Needs Improvement")}</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
            # 4. Generate AI Feedback
            st.markdown("### 🤖 AI Recommendations")
            with st.chat_message("assistant"):
                feedback = generate_ai_feedback(api_key, score, resume_text, jd_text)
                st.markdown(feedback)
                
    else:

        st.warning("Please upload a resume and paste a job description first!")

