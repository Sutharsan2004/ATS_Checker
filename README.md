# 🚀 Smart ATS Resume Analyst

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](YOUR_STREAMLIT_LINK_HERE)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![LangChain](https://img.shields.io/badge/LangChain-🦜-green)
![Groq](https://img.shields.io/badge/Groq-Llama3-orange)

**Live Demo:** [Click here to use the App]((https://sutharsan2004-ats-checker-ats-checker-with-ai-suggestion-sezexm.streamlit.app/))

## 💡 Overview
Applying for jobs can feel like sending applications into a black hole. This **AI-Powered ATS (Application Tracking System) Analyst** helps job seekers optimize their resumes by comparing them against specific job descriptions. 

Unlike simple keyword matchers, this tool uses **Semantic Analysis** and **LLMs** to provide a match score and actionable feedback.

## ⚙️ How It Works
1.  **Resume Parsing:** Extracts text from PDF resumes using `PyPDF2`.
2.  **Semantic Matching:** Uses `Sentence-Transformers` (HuggingFace) to convert the resume and job description into vector embeddings.
3.  **Similarity Scoring:** Calculates the **Cosine Similarity** between the two vectors to determine how closely the resume aligns with the job requirements.
4.  **AI Analysis:** Uses **Llama-3.3-70b** (via Groq) to act as a Senior Recruiter, providing specific feedback on missing keywords and formatting.

## 🛠️ Tech Stack
* **Frontend:** Streamlit
* **LLM Engine:** Groq API (Llama-3.3-70b)
* **Orchestration:** LangChain
* **Embeddings:** `all-MiniLm-L6-v2` (Sentence Transformers)
* **Math/Logic:** Scikit-learn (Cosine Similarity), NumPy

## 🚀 Run Locally
If you want to run this project on your own machine:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
    cd YOUR_REPO_NAME
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the app:**
    ```bash
    streamlit run app.py
    ```

