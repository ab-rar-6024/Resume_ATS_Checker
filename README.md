# 🎯 Resume ATS Score Checker

A complete NLU-powered tool that analyzes your resume against a job description
and produces a detailed ATS compatibility score — helping you get past automated
resume filters and into human hands.

---

## 📁 Project Structure

```
resume_ats_checker/
│
├── app.py                        ← Streamlit UI (main entry point)
│
├── parser/
│   ├── __init__.py
│   └── resume_parser.py          ← PDF/DOCX/TXT extraction + section detection
│
├── nlp/
│   ├── __init__.py
│   ├── keyword_extractor.py      ← NER, TF-IDF, skill taxonomy matching
│   └── similarity.py             ← SBERT semantic similarity scoring
│
├── scorer/
│   ├── __init__.py
│   └── ats_scorer.py             ← Weighted scoring engine + suggestions
│
├── requirements.txt
└── README.md
```

---

## 🧠 How the Score Works

| Component              | Weight | What it measures                                    |
|------------------------|--------|-----------------------------------------------------|
| 🤖 Semantic Similarity | 30%    | Overall meaning alignment (SBERT cosine similarity) |
| 🔧 Technical Keywords  | 30%    | Hard skill/tool overlap with JD                     |
| 💬 Soft Skills         | 10%    | Communication, leadership, agile, etc.              |
| 📜 Certifications      | 10%    | AWS, PMP, CISSP, etc.                               |
| 📅 Experience          | 10%    | Years of experience vs JD requirement               |
| 🎓 Education           | 10%    | Degree level vs JD requirement                      |

**Score interpretation:**
- 85–100 → **A** — Excellent Match 🟢
- 70–84  → **B** — Good Match 🟡
- 55–69  → **C** — Fair Match 🟠
- 40–54  → **D** — Weak Match 🔴
- 0–39   → **F** — Poor Match ⛔

---

## 🛠️ Step-by-Step Setup Guide

### Step 1: Prerequisites

Make sure you have Python 3.9+ installed:
```bash
python --version
```

Install pip if not present:
```bash
python -m ensurepip --upgrade
```

---

### Step 2: Clone / Download the Project

If using git:
```bash
git clone <your-repo-url>
cd resume_ats_checker
```

Or just download and extract the zip, then open a terminal in the folder.

---

### Step 3: Create a Virtual Environment

```bash
# Create venv
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

---

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `streamlit` — UI framework
- `sentence-transformers` — SBERT for semantic similarity
- `scikit-learn` — TF-IDF vectorizer
- `pdfplumber` — PDF text extraction
- `python-docx` — Word document parsing
- `spacy` — NLP pipeline (optional, used for advanced NER)
- `pandas`, `numpy` — Data utilities

> ⚠️ `sentence-transformers` will download the `all-MiniLM-L6-v2` model
> (~80MB) on first run. You need an internet connection for this.

---

### Step 5: (Optional) Download spaCy Model

For enhanced Named Entity Recognition:
```bash
python -m spacy download en_core_web_sm
```

---

### Step 6: Run the App

```bash
streamlit run app.py
```

Your browser will automatically open at `http://localhost:8501`

---

### Step 7: Using the App

1. **Upload Resume** — Click "Browse files" and select your PDF, DOCX, or TXT resume
2. **Paste Job Description** — Copy the full JD from LinkedIn, Indeed, or the company site
3. **Click "Analyze My Resume"** — Wait 10–30 seconds (first run downloads the AI model)
4. **Review Results:**
   - See your overall ATS score (0–100)
   - Check which keywords you're missing
   - Read the semantic similarity breakdown
   - Follow the actionable suggestions

---

## 🔧 Customization Guide

### Add More Skills to the Taxonomy

Edit `nlp/keyword_extractor.py` and add to `TECH_SKILLS`, `SOFT_SKILLS`, or `CERTIFICATIONS`:

```python
TECH_SKILLS = {
    ...
    "your_new_skill",    # ← Add here
}
```

### Change the Scoring Weights

Edit `scorer/ats_scorer.py`:

```python
WEIGHTS = {
    "semantic_similarity": 0.30,   # ← Adjust these
    "technical_keywords":  0.30,
    "soft_skills":         0.10,
    "certifications":      0.10,
    "experience":          0.10,
    "education":           0.10,
}
# Must sum to 1.0
```

### Use a Better Similarity Model

Edit `nlp/similarity.py`, change:

```python
model = SentenceTransformer("all-MiniLM-L6-v2")
# Options (better but slower):
# "all-mpnet-base-v2"
# "multi-qa-mpnet-base-dot-v1"
```

### Add LLM-Powered Rewrite Suggestions

In `scorer/ats_scorer.py`, add a function that calls the Claude or OpenAI API
with the missing keywords and resume bullets to generate rewrite suggestions.

```python
import anthropic

def generate_rewrite_suggestions(bullet_point: str, missing_keywords: list) -> str:
    client = anthropic.Anthropic()
    prompt = f"""
    Rewrite this resume bullet point to naturally include these skills: {missing_keywords}
    Original: {bullet_point}
    Keep it concise and action-verb-led.
    """
    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text
```

---

## 🚀 Deployment Options

### Deploy on Streamlit Cloud (Free)

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set the main file to `app.py`
5. Deploy!

### Deploy with Docker

Create a `Dockerfile`:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Then:
```bash
docker build -t ats-checker .
docker run -p 8501:8501 ats-checker
```

---

## 🧪 Testing with Sample Data

To quickly test without uploading files, add this to `app.py` below imports:

```python
SAMPLE_RESUME = """
John Doe | john@email.com | LinkedIn: linkedin.com/in/johndoe | GitHub: github.com/johndoe

SUMMARY
Experienced Python developer with 4 years of experience in machine learning and data engineering.

SKILLS
Python, TensorFlow, PyTorch, SQL, Docker, AWS, Git, Agile, Scrum

EXPERIENCE
Data Scientist - TechCorp (2020-2024)
- Built ML models improving recommendation accuracy by 35%
- Deployed pipelines using Docker and AWS
- Led cross-functional team of 5 engineers

EDUCATION
B.Tech in Computer Science, MIT (2020)
"""

SAMPLE_JD = """
We are looking for a Senior Data Scientist with 3+ years of experience.
Requirements: Python, TensorFlow, SQL, Docker, AWS, communication skills, teamwork.
Bachelor's degree required.
"""
```

---

## 📌 Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` in your venv |
| PDF not parsing correctly | Try converting to DOCX or TXT first |
| Slow first run | SBERT model downloads ~80MB on first use |
| Port 8501 in use | Run `streamlit run app.py --server.port 8502` |
| Low score despite good resume | JD may use different terminology — check the semantic tab |

---

## 📚 Technologies Used

| Library | Purpose |
|---|---|
| Streamlit | Interactive web UI |
| SentenceTransformers (SBERT) | Semantic similarity via `all-MiniLM-L6-v2` |
| scikit-learn | TF-IDF vectorization |
| pdfplumber | PDF text extraction |
| python-docx | DOCX parsing |
| pandas | Data manipulation |

---

*Built as an advanced NLU project demonstrating: NER, TF-IDF, semantic embeddings, and scoring engines.*#   R e s u m e _ A T S _ C h e c k e r  
 