"""
app.py — Resume ATS Score Checker
Run with:  streamlit run app.py
"""

import streamlit as st
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from parser.resume_parser import extract_text, detect_sections, extract_contact_info
from nlp.keyword_extractor import full_keyword_analysis
from nlp.similarity import compute_similarity
from scorer.ats_scorer import full_ats_score


# ── Page Config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Resume ATS Score Checker",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .score-circle {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 160px;
        height: 160px;
        border-radius: 50%;
        font-size: 2.8rem;
        font-weight: 800;
        margin: 0 auto;
        color: white;
    }
    .metric-card {
        background: #1e1e2e;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.5rem;
        border-left: 4px solid;
    }
    .pill {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.78rem;
        margin: 2px;
        font-weight: 500;
    }
    .pill-green  { background: #166534; color: #86efac; }
    .pill-red    { background: #7f1d1d; color: #fca5a5; }
    .pill-blue   { background: #1e3a5f; color: #93c5fd; }
    .pill-gray   { background: #374151; color: #d1d5db; }
    .suggestion  { background: #1a1a2e; border-left: 3px solid #6366f1;
                   padding: 0.7rem 1rem; border-radius: 8px; margin: 0.5rem 0; }
    .section-header { font-size: 1.1rem; font-weight: 700; margin-bottom: 0.5rem; }
    div[data-testid="metric-container"] { background: #1e1e2e; border-radius: 10px; padding: 0.8rem; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📄 ATS Score Checker")
    st.markdown("---")
    st.markdown("### How it works")
    st.markdown("""
    1. **Upload** your resume (PDF/DOCX/TXT)
    2. **Paste** the job description
    3. **Analyze** and get your score
    4. **Fix** the gaps and resubmit
    """)
    st.markdown("---")
    st.markdown("### Score Breakdown")
    for label, weight in [
        ("🤖 Semantic Similarity", "30%"),
        ("🔧 Technical Keywords", "30%"),
        ("💬 Soft Skills", "10%"),
        ("📜 Certifications", "10%"),
        ("📅 Experience", "10%"),
        ("🎓 Education", "10%"),
    ]:
        st.markdown(f"**{label}** — `{weight}`")


# ── Main UI ────────────────────────────────────────────────────────────────────

st.title("🎯 Resume ATS Score Checker")
st.markdown("*Analyze how well your resume matches a job description and get actionable feedback.*")
st.markdown("---")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📎 Upload Resume")
    resume_file = st.file_uploader(
        "Drag and drop your resume here",
        type=["pdf", "docx", "txt"],
        help="Supported formats: PDF, DOCX, TXT",
    )
    if resume_file:
        st.success(f"✅ Uploaded: `{resume_file.name}`")

with col2:
    st.subheader("📋 Job Description")
    jd_text = st.text_area(
        "Paste the full job description here",
        height=200,
        placeholder="Paste the job description from LinkedIn, Indeed, etc...",
    )

st.markdown("---")

# ── Analysis Button ────────────────────────────────────────────────────────────

analyze_btn = st.button(
    "🚀 Analyze My Resume",
    type="primary",
    use_container_width=True,
    disabled=(resume_file is None or not jd_text.strip()),
)

if not resume_file:
    st.info("👆 Upload your resume and paste the job description to get started.")

if analyze_btn:
    if not resume_file:
        st.error("Please upload a resume file.")
        st.stop()
    if not jd_text.strip():
        st.error("Please paste the job description.")
        st.stop()

    # ── Run Analysis ───────────────────────────────────────────────────────────

    with st.spinner("🔍 Parsing resume..."):
        file_ext = resume_file.name.rsplit(".", 1)[-1]
        resume_text = extract_text(resume_file, file_ext)
        if resume_text.startswith("Error"):
            st.error(resume_text)
            st.stop()
        sections = detect_sections(resume_text)
        contact_info = extract_contact_info(resume_text)

    with st.spinner("🧠 Running NLP analysis..."):
        kw_analysis = full_keyword_analysis(resume_text, jd_text)

    with st.spinner("⚡ Computing semantic similarity..."):
        similarity_result = compute_similarity(resume_text, jd_text)

    with st.spinner("📊 Computing ATS score..."):
        report = full_ats_score(resume_text, jd_text, kw_analysis, similarity_result)

    # ── Results ────────────────────────────────────────────────────────────────

    st.markdown("---")
    st.markdown("## 📊 Results")

    # Score Circle + Grade
    score = report["final_score"]
    color = report["color"]
    grade = report["grade"]
    label = report["label"]

    top_col1, top_col2, top_col3 = st.columns([1, 1, 2])

    with top_col1:
        st.markdown(
            f'<div class="score-circle" style="background:{color}">'
            f'{score:.0f}<span style="font-size:1rem">%</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="text-align:center;margin-top:0.5rem;font-size:1.2rem;font-weight:700">'
            f'Grade: {grade}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="text-align:center;color:{color};font-weight:600">{label}</div>',
            unsafe_allow_html=True,
        )

    with top_col2:
        comps = report["components"]
        st.metric("🤖 Semantic Match",
                  f"{comps['semantic_similarity']['overall_percent']:.1f}%",
                  help="How semantically similar your resume is to the JD")
        st.metric("🔧 Technical Keywords",
                  f"{comps['technical_keywords']['matched']}/{comps['technical_keywords']['total_required']}",
                  help="Technical skills matched out of JD requirements")

    with top_col3:
        # Component breakdown bar chart
        import pandas as pd
        chart_data = pd.DataFrame({
            "Component": [
                "Semantic Similarity",
                "Technical Keywords",
                "Soft Skills",
                "Certifications",
                "Experience",
                "Education",
            ],
            "Score (%)": [
                comps["semantic_similarity"]["overall_percent"],
                comps["technical_keywords"]["score"] * 100,
                comps["soft_skills"]["score"] * 100,
                comps["certifications"]["score"] * 100,
                comps["experience"]["score"] * 100,
                comps["education"]["score"] * 100,
            ],
        })
        st.bar_chart(chart_data.set_index("Component"), height=220)

    st.markdown("---")

    # ── Detailed Breakdown ─────────────────────────────────────────────────────

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔑 Keywords", "💬 Soft Skills", "📜 Certs & Education", "🤖 Semantic", "📝 Suggestions"
    ])

    # --- Keywords Tab ---
    with tab1:
        k = comps["technical_keywords"]
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("#### ✅ Matched Technical Skills")
            if k["matched_keywords"]:
                pills = " ".join(
                    f'<span class="pill pill-green">{s}</span>'
                    for s in k["matched_keywords"]
                )
                st.markdown(pills, unsafe_allow_html=True)
            else:
                st.markdown("*No matches found.*")

        with col_b:
            st.markdown("#### ❌ Missing Technical Skills")
            if k["missing_keywords"]:
                pills = " ".join(
                    f'<span class="pill pill-red">{s}</span>'
                    for s in sorted(k["missing_keywords"])[:20]
                )
                st.markdown(pills, unsafe_allow_html=True)
            else:
                st.markdown("✅ *All required skills present!*")

        st.markdown("---")
        st.markdown("#### 🔍 TF-IDF Important JD Keywords Missing from Resume")
        if report["tfidf_missing"]:
            pills = " ".join(
                f'<span class="pill pill-gray">{s}</span>'
                for s in report["tfidf_missing"]
            )
            st.markdown(pills, unsafe_allow_html=True)
        else:
            st.markdown("*None — great keyword coverage!*")

    # --- Soft Skills Tab ---
    with tab2:
        s = comps["soft_skills"]
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### ✅ Matched Soft Skills")
            if s["matched_keywords"]:
                pills = " ".join(
                    f'<span class="pill pill-green">{sk}</span>'
                    for sk in s["matched_keywords"]
                )
                st.markdown(pills, unsafe_allow_html=True)
            else:
                st.markdown("*No matches found.*")
        with col_b:
            st.markdown("#### ❌ Missing Soft Skills")
            if s["missing_keywords"]:
                pills = " ".join(
                    f'<span class="pill pill-red">{sk}</span>'
                    for sk in s["missing_keywords"]
                )
                st.markdown(pills, unsafe_allow_html=True)
            else:
                st.markdown("✅ *All soft skills present!*")

    # --- Certs & Education Tab ---
    with tab3:
        c = comps["certifications"]
        e = comps["education"]
        ex = comps["experience"]

        st.markdown("#### 📜 Certifications")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Matched**")
            if c["matched_keywords"]:
                for cert in c["matched_keywords"]:
                    st.markdown(f"✅ {cert}")
            else:
                st.markdown("*None found.*")
        with col_b:
            st.markdown("**Missing**")
            if c["missing_keywords"]:
                for cert in c["missing_keywords"]:
                    st.markdown(f"❌ {cert}")
            else:
                st.markdown("*None required or all present!*")

        st.markdown("---")
        st.markdown("#### 🎓 Education")
        edu_col1, edu_col2 = st.columns(2)
        with edu_col1:
            st.info(f"**Your Resume:** {e['resume_education']}")
        with edu_col2:
            st.info(f"**JD Requirement:** {e['jd_education']}")
        st.markdown(e["note"])

        st.markdown("---")
        st.markdown("#### 📅 Experience")
        ex_col1, ex_col2 = st.columns(2)
        with ex_col1:
            st.info(f"**Your Experience:** {ex['resume_years']} years")
        with ex_col2:
            st.info(f"**JD Requires:** {ex['jd_required_years']} years")
        st.markdown(ex.get("note", ""))

    # --- Semantic Tab ---
    with tab4:
        sem = comps["semantic_similarity"]
        st.markdown(f"**Method:** `{sem['method']}`")
        st.markdown(f"**Overall Semantic Similarity:** `{sem['overall_percent']:.2f}%`")
        st.progress(min(sem["overall_score"], 1.0))

        if sem.get("top_sentence_matches"):
            st.markdown("---")
            st.markdown("#### 🔗 Top Matching Passages")
            st.markdown("*JD requirements that your resume covers best:*")
            for match in sem["top_sentence_matches"][:5]:
                score_pct = match["score"] * 100
                color_badge = "#22c55e" if score_pct > 70 else "#eab308" if score_pct > 50 else "#ef4444"
                st.markdown(
                    f'<div class="metric-card" style="border-color:{color_badge}">'
                    f'<div style="color:#aaa;font-size:0.8rem">JD: {match["jd_sentence"]}</div>'
                    f'<div style="color:#ccc;font-size:0.8rem;margin-top:4px">Resume: {match["best_resume_match"]}</div>'
                    f'<div style="color:{color_badge};font-weight:700;margin-top:4px">Score: {score_pct:.1f}%</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # --- Suggestions Tab ---
    with tab5:
        st.markdown("### 💡 Actionable Improvement Suggestions")
        for suggestion in report["suggestions"]:
            st.markdown(
                f'<div class="suggestion">{suggestion}</div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown("### 📬 Contact Info Detected in Resume")
        info = contact_info
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Email", info["email"] or "❌ Not found")
        c2.metric("Phone", info["phone"] or "❌ Not found")
        c3.metric("LinkedIn", "✅ Found" if info["linkedin"] else "❌ Not found")
        c4.metric("GitHub", "✅ Found" if info["github"] else "❌ Not found")

        st.markdown("---")
        st.markdown("### 📂 Detected Resume Sections")
        detected = list(sections.keys())
        important = ["experience", "skills", "education", "summary", "projects", "certifications"]
        for section in important:
            if section in detected:
                st.markdown(f"✅ **{section.title()}** — Found")
            else:
                st.markdown(f"❌ **{section.title()}** — Not detected")


# ── Footer ─────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    '<div style="text-align:center;color:#666;font-size:0.8rem">'
    'Resume ATS Score Checker | Built with Streamlit, spaCy, SentenceTransformers'
    '</div>',
    unsafe_allow_html=True,
)