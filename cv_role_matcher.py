import streamlit as st
import pymupdf
import pandas as pd
from openai import OpenAI
import tempfile
import re

st.title("🧠 CV-to-Job Matcher")

# API Key input
api_key = st.text_input("🔐 Enter your OpenAI API Key", type="password")
client = OpenAI(api_key=api_key) if api_key else None

# Upload job descriptions
jd_files = st.file_uploader("📄 Upload Job Descriptions (TXT or PDF)", type=["txt", "pdf"], accept_multiple_files=True)

# Upload CVs
cv_files = st.file_uploader("📎 Upload Candidate CVs (PDF)", type="pdf", accept_multiple_files=True)

if st.button("▶️ Run Matching") and jd_files and cv_files and client:
    st.info("Processing...")

    # Read JDs
    jd_texts = {}
    for jd_file in jd_files:
        if jd_file.name.endswith(".pdf"):
            with fitz.open(stream=jd_file.read(), filetype="pdf") as doc:
                text = "\n".join(page.get_text() for page in doc)
        else:
            text = jd_file.read().decode("utf-8")
        jd_texts[jd_file.name] = text

    # Read CVs
    cv_texts = {}
    for cv_file in cv_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(cv_file.read())
            tmp.flush()
            doc = fitz.open(tmp.name)
            text = "\n".join(page.get_text() for page in doc)
            doc.close()
        cv_texts[cv_file.name] = text

    # Run matchings
    match_data = []
    for cv_name, cv_text in cv_texts.items():
        row = {"CV": cv_name}
        best_score = -1
        best_role = None

        for jd_name, jd_text in jd_texts.items():
            prompt = f"""
You are an HR assistant. Compare the following CV and job description. Return a match percentage (0–100) and a short explanation.

Job Description:
{jd_text}

CV:
{cv_text}

Respond in this format only:
Match Percentage: XX%
Explanation: ...
"""
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful HR assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=500
            )

            result = response.choices[0].message.content
            match = re.search(r"Match Percentage:\s*(\d+)", result)
            score = int(match.group(1)) if match else 0
            row[jd_name] = score

            if score > best_score:
                best_score = score
                best_role = jd_name

        row["Best Match"] = best_role
        match_data.append(row)

    df = pd.DataFrame(match_data)
        # Determine best CV for each job description
    top_candidates = []
    for jd_name in jd_texts.keys():
        if jd_name in df.columns:
            top_row = df.loc[df[jd_name].idxmax()]
            top_candidates.append({"Job": jd_name, "Best Candidate": top_row["CV"], "Score": top_row[jd_name]})

        st.subheader("🌟 Best Candidate per Job")
    st.dataframe(df_top)
    st.download_button("📥 Download Top Candidates", df_top.to_csv(index=False), "top_candidates_per_job.csv")

    
    df_top = pd.DataFrame(top_candidates)

    st.success("✅ Matching complete!")
    st.dataframe(df)

    st.download_button("📥 Download Match Results", df.to_csv(index=False), "cv_match_matrix.csv")
