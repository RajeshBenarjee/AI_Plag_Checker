import nltk

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

from utils.report_generator    import generate_pdf_report
from utils.pdf_parser          import extract_text_from_pdf
from utils.text_processing     import sentence_tokenize
from utils.diff_viewer         import build_diff_html
from utils.batch_processor     import BatchProcessor
from utils.trust_score         import compute_trust_score
from utils.auth_manager        import register, login, get_all_users
from utils.submission_tracker  import (save_submission, get_student_history,
                                       get_all_submissions, get_dashboard_stats)
from utils.section_heatmap     import build_section_heatmap, build_sentence_timeline
from model.embedding_model     import embed_sentences
from model.plagiarism_engine   import PlagiarismEngine
from model.internet_detector   import InternetDetector
from model.cross_student_detector import CrossStudentDetector
from model.style_fingerprint   import StyleFingerprint
from model.anomaly_detector    import detect_anomalies
from model.paraphrase_classifier import batch_classify

load_dotenv()
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

# PAGE CONFIG
st.set_page_config(
    page_title="AI Plagiarism Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# SESSION STATE INIT
if "logged_in"  not in st.session_state: st.session_state.logged_in  = False
if "username"   not in st.session_state: st.session_state.username   = ""
if "role"       not in st.session_state: st.session_state.role       = ""
if "auth_page"  not in st.session_state: st.session_state.auth_page  = "login"


# ── AUTH WALL ── show login/register before anything else
if not st.session_state.logged_in:

    st.title("🔍 AI Plagiarism Detection System")
    st.markdown("---")

    col_left, col_center, col_right = st.columns([1, 2, 1])

    with col_center:

        tab_login, tab_register = st.tabs(["🔐 Login", "📝 Register"])

        # ── LOGIN ──
        with tab_login:
            st.subheader("Welcome Back")
            login_user = st.text_input("Username", key="login_user")
            login_pass = st.text_input("Password", type="password", key="login_pass")

            if st.button("Login", use_container_width=True, key="btn_login"):
                if login_user and login_pass:
                    ok, msg, user_info = login(login_user, login_pass)
                    if ok:
                        st.session_state.logged_in = True
                        st.session_state.username  = user_info["username"]
                        st.session_state.role      = user_info["role"]
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)
                else:
                    st.warning("Please enter username and password.")

        # ── REGISTER ──
        with tab_register:
            st.subheader("Create Account")
            reg_user  = st.text_input("Username",        key="reg_user")
            reg_pass  = st.text_input("Password",        type="password", key="reg_pass")
            reg_pass2 = st.text_input("Confirm Password",type="password", key="reg_pass2")
            reg_role  = st.selectbox("Role", ["student", "teacher"], key="reg_role")

            if st.button("Register", use_container_width=True, key="btn_register"):
                if reg_pass != reg_pass2:
                    st.error("Passwords do not match.")
                elif reg_user and reg_pass:
                    ok, msg = register(reg_user, reg_pass, reg_role)
                    if ok:
                        st.success(msg + " Please login.")
                    else:
                        st.error(msg)
                else:
                    st.warning("Please fill all fields.")

    st.stop()  

#  SIDEBAR (shown after login)
with st.sidebar:
    st.markdown(f"### 👤 {st.session_state.username}")
    st.markdown(f"Role: `{st.session_state.role}`")
    st.divider()

    if st.session_state.role == "teacher":
        page = st.radio("Navigate", [
            "🏠 Dashboard",
            "📚 Batch Upload",
            "👥 Compare Students",
            "📝 Check Single",
            "📋 All Submissions"
        ])
    else:
        page = st.radio("Navigate", [
            "📝 Check Assignment",
            "📈 My History"
        ])

    st.divider()

    lang_mode = st.selectbox(
        "🌐 Language Mode",
        ["English", "Multilingual (Telugu, Tamil, 50+ languages)"]
    )

    if st.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.session_state.username  = ""
        st.session_state.role      = ""
        st.rerun()


model_type = "multilingual" if "Multilingual" in lang_mode else "english"


# LOAD ENGINES (cached)
@st.cache_resource
def load_engines(mt):
    engine           = PlagiarismEngine(model_type=mt)
    internet_det     = InternetDetector(SERPAPI_KEY)
    student_det      = CrossStudentDetector()
    batch_proc       = BatchProcessor(engine)
    style_fp         = StyleFingerprint()
    return engine, internet_det, student_det, batch_proc, style_fp

engine, internet_detector, student_detector, batch_processor, style_fp = load_engines(model_type)


#  TEACHER: DASHBOARD 
if st.session_state.role == "teacher" and page == "🏠 Dashboard":

    st.title("🏠 Teacher Dashboard")

    stats = get_dashboard_stats()

    # ── KPI Cards ──
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("📄 Total Submissions",   stats["total_submissions"])
    k2.metric("📊 Avg Plagiarism",      f"{stats['avg_plagiarism_score']}%")
    k3.metric("👨‍🎓 Students Tracked",   len(stats["student_stats"]))
    k4.metric("🚨 High Risk Students",
              len([s for s in stats["student_stats"].values() if s["avg_plagiarism"] > 60]))

    st.divider()

    # ── Top Flagged Students ──
    st.subheader("🚨 Top Flagged Students")

    if stats["top_flagged"]:
        df_flagged = pd.DataFrame(stats["top_flagged"])
        df_flagged.columns = ["Username","Submissions","Avg Plagiarism %",
                               "Last Submission","Highest Score"]

        def color_score(val):
            if isinstance(val, float):
                if val > 60: return "background-color:#ffcccc"
                elif val > 30: return "background-color:#fff9c4"
                else: return "background-color:#c8e6c9"
            return ""

        st.dataframe(
            df_flagged.style.applymap(color_score, subset=["Avg Plagiarism %"]),
            use_container_width=True
        )
    else:
        st.info("No submissions yet.")

    # ── Trend Chart ──
    st.subheader("📈 Plagiarism Trends Over Time")

    all_subs = get_all_submissions()
    all_records = []

    for user, records in all_subs.items():
        for r in records:
            all_records.append({
                "student":   user,
                "timestamp": r["timestamp"][:10],
                "score":     r["percentage"]
            })

    if all_records:
        df_trend = pd.DataFrame(all_records)
        df_trend["timestamp"] = pd.to_datetime(df_trend["timestamp"])
        df_trend_grouped = df_trend.groupby("timestamp")["score"].mean().reset_index()

        fig_trend = px.line(
            df_trend_grouped,
            x="timestamp", y="score",
            title="Average Plagiarism Score Over Time",
            labels={"score": "Avg Plagiarism %", "timestamp": "Date"},
            markers=True
        )
        fig_trend.add_hline(y=60, line_dash="dash", line_color="red",
                            annotation_text="High Risk Threshold")
        st.plotly_chart(fig_trend, use_container_width=True)

    # ── Recent Submissions ──
    st.subheader("🕐 Recent Submissions")

    if stats["recent_submissions"]:
        df_recent = pd.DataFrame(stats["recent_submissions"])
        st.dataframe(df_recent[["username","filename","percentage",
                                 "trust_verdict","timestamp"]],
                     use_container_width=True)
    else:
        st.info("No recent submissions.")

    # ── All Users ──
    st.subheader("👥 Registered Users")
    all_users = get_all_users()
    if all_users:
        st.dataframe(pd.DataFrame(all_users), use_container_width=True)


#  TEACHER: ALL SUBMISSIONS   
elif st.session_state.role == "teacher" and page == "📋 All Submissions":

    st.title("📋 All Student Submissions")

    all_subs = get_all_submissions()

    if not all_subs:
        st.info("No submissions yet.")
    else:
        student_filter = st.selectbox("Filter by student", ["All"] + list(all_subs.keys()))

        records = []
        for user, subs in all_subs.items():
            if student_filter != "All" and user != student_filter:
                continue
            for s in subs:
                records.append({**s, "student": user})

        if records:
            df = pd.DataFrame(records)
            st.dataframe(df, use_container_width=True)


#  STUDENT: MY HISTORY 
elif page == "📈 My History":

    st.title(f"📈 My Submission History — {st.session_state.username}")

    history = get_student_history(st.session_state.username)

    if not history:
        st.info("You haven't submitted any assignments yet.")
    else:
        df_hist = pd.DataFrame(history)

        # ── Timeline chart ──
        st.subheader("📊 Plagiarism Score Over Time")

        fig_hist = px.line(
            df_hist,
            x="timestamp", y="percentage",
            title="Your Plagiarism % per Submission",
            markers=True,
            color_discrete_sequence=["#e74c3c"]
        )
        fig_hist.add_hline(y=60, line_dash="dash", line_color="red",
                            annotation_text="High Risk")
        fig_hist.add_hline(y=30, line_dash="dash", line_color="orange",
                            annotation_text="Suspicious")
        st.plotly_chart(fig_hist, use_container_width=True)

        #  Trust score timeline 
        st.subheader("🛡️ Trust Score Over Time")

        fig_trust = px.line(
            df_hist,
            x="timestamp", y="trust_score",
            title="Trust Score per Submission",
            markers=True,
            color_discrete_sequence=["#3498db"]
        )
        st.plotly_chart(fig_trust, use_container_width=True)

        #  Raw table 
        st.subheader("📋 Submission Log")
        st.dataframe(df_hist, use_container_width=True)


#  TEACHER: BATCH UPLOAD 
elif (st.session_state.role == "teacher" and page == "📚 Batch Upload"):

    st.title("📚 Batch Upload — Full Class Analysis")
    st.caption("Upload all student PDFs at once.")

    batch_files = st.file_uploader(
        "Upload All Student PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        key="batch_upload"
    )

    if st.button("🔍 Analyze Full Class") and batch_files:

        with st.spinner("Processing all submissions..."):
            summaries, sim_matrix = batch_processor.process_batch(batch_files)

        st.success(f"✅ Analyzed {len(summaries)} submissions")

        df_summary = pd.DataFrame([{
            "Student":       s["filename"],
            "Total":         s["total"],
            "Plagiarized":   s["plagiarized"],
            "Plagiarism %":  s["percentage"]
        } for s in summaries])

        def color_row(val):
            if val > 60:   return "background-color:#ffcccc"
            elif val > 30: return "background-color:#fff9c4"
            else:          return "background-color:#c8e6c9"

        st.dataframe(
            df_summary.style.applymap(color_row, subset=["Plagiarism %"]),
            use_container_width=True
        )

        st.subheader("🗺️ Student-to-Student Similarity Matrix")

        filenames    = list(sim_matrix.keys())
        matrix_data  = [[sim_matrix[r][c] for c in filenames] for r in filenames]

        fig_heat = px.imshow(
            matrix_data, x=filenames, y=filenames,
            color_continuous_scale="RdYlGn_r",
            zmin=0, zmax=1,
            title="Student-to-Student Similarity Matrix",
            text_auto=".2f"
        )
        fig_heat.update_layout(height=500)
        st.plotly_chart(fig_heat, use_container_width=True)

        st.subheader("⚠️ Top 3 Most Suspicious Pairs")

        pairs = []
        for i, fnA in enumerate(filenames):
            for j, fnB in enumerate(filenames):
                if j <= i: continue
                pairs.append((fnA, fnB, sim_matrix[fnA][fnB]))

        for fnA, fnB, score in sorted(pairs, key=lambda x: x[2], reverse=True)[:3]:
            st.warning(f"**{fnA}** ↔ **{fnB}** — Similarity: `{score:.2%}`")


#  TEACHER: COMPARE STUDENTS
elif (st.session_state.role == "teacher" and page == "👥 Compare Students"):

    st.title("👥 Compare Two Student Assignments")

    colA, colB = st.columns(2)

    with colA:
        uploaded_file1 = st.file_uploader("Upload Student A", type=["pdf"], key="studentA")
    with colB:
        uploaded_file2 = st.file_uploader("Upload Student B", type=["pdf"], key="studentB")

    text1 = extract_text_from_pdf(uploaded_file1) if uploaded_file1 else None
    text2 = extract_text_from_pdf(uploaded_file2) if uploaded_file2 else None

    if st.button("⚖️ Compare") and text1 and text2:

        sentencesA = sentence_tokenize(text1)
        sentencesB = sentence_tokenize(text2)

        embA = embed_sentences(sentencesA, model_type=model_type)
        embB = embed_sentences(sentencesB, model_type=model_type)

        sim_matrix_ab = cosine_similarity(embA, embB)
        matches = []

        for i, row in enumerate(sim_matrix_ab):
            best_idx   = row.argmax()
            best_score = row[best_idx]
            if best_score >= 0.75:
                matches.append({
                    "sentenceA": sentencesA[i],
                    "sentenceB": sentencesB[best_idx],
                    "score":     float(best_score)
                })

        overall = (len(matches) / len(sentencesA)) * 100 if sentencesA else 0

        st.metric("Overall Similarity", f"{overall:.2f}%")

        st.subheader("🔗 Matched Sentences")

        for m in matches:
            with st.container():
                c1, c2 = st.columns([3, 1])
                with c1:
                    st.markdown(f"**Student A:** {m['sentenceA']}")
                    st.markdown(f"*Student B:* {m['sentenceB']}")
                with c2:
                    st.progress(min(max(m["score"], 0.0), 1.0))
                st.warning(f"Similarity: {round(m['score'], 3)}")

        st.subheader("📄 Side-by-Side Diff View")
        diff_html = build_diff_html(sentencesA, sentencesB, matches)
        st.markdown(diff_html, unsafe_allow_html=True)


#  SINGLE DOCUMENT CHECK (both teacher + student) 
elif page in ("📝 Check Single", "📝 Check Assignment"):

    username = st.session_state.username
    role     = st.session_state.role

    st.title("📝 Check Assignment for Plagiarism")

    text_input    = st.text_area("Paste text here")
    uploaded_file = st.file_uploader("Or upload PDF/TXT", type=["pdf","txt"],
                                     key="single_upload")

    text     = None
    filename = "pasted_text"

    if uploaded_file:
        text     = extract_text_from_pdf(uploaded_file)
        filename = uploaded_file.name
    elif text_input:
        text = text_input

    # Checkboxes
    st.subheader("⚙️ Detection Features")
    c1, c2 = st.columns(2)
    with c1:
        exact       = st.checkbox("Exact copied text only")
        paraphrased = st.checkbox("Paraphrased sentences only")
    with c2:
        section_view = st.checkbox("Section-wise analysis")
        highlight    = st.checkbox("Highlight document")

    if st.button("🚀 Check Plagiarism") and text:

        #  Run detection 
        with st.spinner("Running AI plagiarism detection..."):
            results, percentage, total, plagiarized, cited_sentences = engine.detect(text)

        st.success("✅ Analysis Completed")

        #  Cited sentences 
        if cited_sentences:
            with st.expander(f"📚 {len(cited_sentences)} Cited Sentences Excluded"):
                for s in cited_sentences:
                    st.write("✅", s)

        #  Internet check 
        internet_hits = []
        with st.spinner("🌐 Scanning internet..."):
            for r in results:
                if r["score"] < 0.40:
                    continue
                web_score, web_match, web_source = internet_detector.detect(r["sentence"])
                if web_source:
                    internet_hits.append({
                        "sentence":   r["sentence"],
                        "web_score":  web_score,
                        "web_match":  web_match,
                        "web_source": web_source
                    })

        #  Cross student 
        student_results = student_detector.detect_similarity(text)

        #  Trust score 
        trust = compute_trust_score(percentage, results, internet_hits, student_results)

        #  Trust Banner 
        st.markdown(
            f"""
            <div style='background:{trust["color"]};padding:20px;border-radius:10px;
                        text-align:center;margin-bottom:20px;'>
              <span style='font-size:36px;'>{trust["emoji"]}</span>
              <h2 style='color:white;margin:5px 0;'>{trust["verdict"]}</h2>
              <p style='color:white;font-size:18px;margin:0;'>
                Trust Score: {trust["score"]}%
              </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        b = trust["breakdown"]
        t1,t2,t3,t4 = st.columns(4)
        t1.metric("Base Score",       f"{b['base']}%")
        t2.metric("Internet Boost",   f"+{b['internet_boost']}")
        t3.metric("Student Boost",    f"+{b['student_boost']}")
        t4.metric("Confidence Boost", f"+{b['confidence_boost']}")

        #  Gauge 
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=percentage,
            title={"text": "Raw Plagiarism Score"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar":  {"color": "#2c3e50"},
                "steps": [
                    {"range": [0,  30],  "color": "#27ae60"},
                    {"range": [30, 60],  "color": "#f1c40f"},
                    {"range": [60, 80],  "color": "#e67e22"},
                    {"range": [80, 100], "color": "#c0392b"}
                ]
            }
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)

        m1, m2 = st.columns(2)
        m1.metric("Total Sentences",   total)
        m2.metric("Flagged Sentences", plagiarized)

        #  Section Heatmap 
        st.subheader("🗺️ Plagiarism by Document Section")
        fig_section = build_section_heatmap(results)
        st.plotly_chart(fig_section, use_container_width=True)

        fig_timeline = build_sentence_timeline(results)
        st.plotly_chart(fig_timeline, use_container_width=True)

        #  Highlighted document 
        if highlight:
            st.subheader("🖍️ Highlighted Document")
            st.caption("🔴 Copied  🟠 Paraphrased  🟡 Suspicious  🟢 Original")
            html = ""
            for r in results:
                s = r["sentence"]
                if r["score"] >= 0.90:   color = "#ffcccc"
                elif r["score"] >= 0.70: color = "#ffe0b2"
                elif r["score"] >= 0.50: color = "#fff9c4"
                else:                    color = "#c8e6c9"
                html += (f"<span style='background:{color};padding:3px 5px;"
                         f"border-radius:3px;margin:2px;display:inline;'>{s}</span> ")
            st.markdown(html, unsafe_allow_html=True)

        #  Section-wise breakdown 
        if section_view:
            st.subheader("📑 Section-wise Analysis")
            n = len(results)
            intro = results[:max(1, int(n*0.20))]
            body  = results[max(1,int(n*0.20)):max(2,int(n*0.80))]
            concl = results[max(2, int(n*0.80)):]
            for sname, sresults in [("📖 Introduction",intro),
                                     ("📄 Body",body),
                                     ("📌 Conclusion",concl)]:
                flagged = len([r for r in sresults if r["score"] >= 0.70])
                pct = (flagged/len(sresults)*100) if sresults else 0
                st.markdown(f"**{sname}** — {flagged}/{len(sresults)} flagged ({pct:.1f}%)")

        #  Paraphrase classifier 
        st.subheader("🔄 Paraphrase Analysis")
        st.caption("Cross-encoder model checks if flagged sentences are smart rewrites.")

        flagged_pairs = [
            (r["sentence"], r["match"])
            for r in results if r["score"] >= 0.65 and r["match"] != "No strong dataset match found"
        ]

        if flagged_pairs:
            with st.spinner("Running paraphrase classifier..."):
                paraphrase_results = batch_classify(flagged_pairs)

            for (sent, match), pr in zip(flagged_pairs, paraphrase_results):
                with st.container():
                    p1, p2 = st.columns([3,1])
                    with p1:
                        st.markdown(f"**Student:** {sent}")
                        st.markdown(f"*Source:* {match}")
                    with p2:
                        st.progress(pr["cross_score"])
                        st.caption(f"{pr['paraphrase_type']}")

                    if pr["is_paraphrase"]:
                        st.error(f"🔄 {pr['paraphrase_type']} — confidence: {pr['confidence']}")
                    else:
                        st.success(f"✅ {pr['paraphrase_type']}")
                    st.divider()
        else:
            st.success("✅ No paraphrase candidates found.")

        #  Sentence analysis cards 
        st.subheader("🔎 Sentence Analysis")

        filtered = results
        if exact:
            filtered = [r for r in filtered if r["type"]=="Exact" and r["score"]>=0.85]
        if paraphrased:
            filtered = [r for r in filtered if r["type"]=="Semantic" and r["score"]>=0.70]

        for r in filtered:
            with st.container():
                c1, c2 = st.columns([3,1])
                with c1:
                    st.markdown(f"**Student:** {r['sentence']}")
                    st.markdown(f"*Match:* {r['match']}")
                with c2:
                    st.progress(min(max(r["score"],0.0),1.0))
                    st.caption(f"Score: {round(r['score'],3)}")
                st.write("Type:", r["type"])
                if r["score"] >= 0.9:   st.error("🔴 High Plagiarism")
                elif r["score"] >= 0.8: st.warning("🟠 Strong Similarity")
                elif r["score"] >= 0.7: st.info("🟡 Possible Paraphrasing")
                else:                   st.success("🟢 Clean")
                st.divider()

        #  Internet results 
        st.subheader("🌐 Internet Source Matches")
        if internet_hits:
            for hit in internet_hits:
                with st.container():
                    ca, cb = st.columns([3,1])
                    with ca:
                        st.markdown(f"**Student:** {hit['sentence']}")
                        if hit["web_match"]:
                            st.markdown(f"*Web Match:* {hit['web_match']}")
                        if hit["web_source"]:
                            st.markdown(f"🔗 [{hit['web_source']}]({hit['web_source']})")
                    with cb:
                        st.progress(min(max(hit["web_score"],0.0),1.0))
                        st.caption(f"{round(hit['web_score']*100)}% match")
                    if hit["web_score"] >= 0.85:   st.error(f"🔴 High: {round(hit['web_score'],3)}")
                    elif hit["web_score"] >= 0.65: st.warning(f"🟠 Moderate: {round(hit['web_score'],3)}")
                    else:                          st.info(f"🟡 Low: {round(hit['web_score'],3)}")
                    st.divider()
        else:
            st.success("✅ No significant internet matches found.")

        #  Cross student 
        st.subheader("👥 Cross-Student Plagiarism")
        if student_results:
            for r in student_results:
                st.markdown(f"**Current:** {r['sentence']}")
                st.markdown(f"*Previous:* {r['matched']}")
                st.warning(f"Similarity: {r['similarity']}")
                st.divider()
        else:
            st.success("✅ No matches with previous student submissions.")

        student_detector.add_submission(text)

        #  Anomaly detection 
        st.subheader("🚨 Anomaly Detection")
        history = get_student_history(username)

        if len(history) >= 2:
            anomaly = detect_anomalies(text, history)
            if anomaly["verdict"] == "🔴 High Risk":
                st.error(f"{anomaly['verdict']}")
            elif anomaly["verdict"] == "🟡 Suspicious":
                st.warning(f"{anomaly['verdict']}")
            else:
                st.success(f"{anomaly['verdict']}")

            for flag in anomaly["flags"]:
                st.write(flag)
        else:
            st.info("ℹ️ Need at least 2 past submissions to run anomaly detection.")

        # ── Writing style fingerprint ──
        st.subheader("✍️ Writing Style Fingerprint")
        fingerprint = style_fp.analyze(text)

        if fingerprint["verdict"] == "Possible Multiple Authors":
            st.error("⚠️ Possible Multiple Authors / Ghostwriting Detected")
        elif fingerprint["verdict"] == "Consistent Writing Style":
            st.success("✅ Consistent Writing Style")
        else:
            st.info(f"ℹ️ {fingerprint['verdict']}")

        if fingerprint.get("sections"):
            features = ["avg_sentence_length","vocab_richness","avg_word_length",
                        "punctuation_density","stopword_ratio"]

            fig_radar = go.Figure()
            for sname in ["intro","body","conclusion"]:
                sec = fingerprint["sections"].get(sname, {})
                vals = [sec.get(f,0) for f in features]
                vals += vals[:1]
                fig_radar.add_trace(go.Scatterpolar(
                    r=vals, theta=features+[features[0]],
                    fill="toself", name=sname.capitalize()
                ))
            fig_radar.update_layout(
                polar={"radialaxis":{"visible":True}},
                title="Writing Style per Section", showlegend=True
            )
            st.plotly_chart(fig_radar, use_container_width=True)

            d = fingerprint["distances"]
            d1,d2,d3 = st.columns(3)
            d1.metric("Intro vs Body",        d.get("intro_vs_body",0))
            d2.metric("Body vs Conclusion",   d.get("body_vs_conclusion",0))
            d3.metric("Intro vs Conclusion",  d.get("intro_vs_conclusion",0))

        #  Save submission 
        save_submission(
            username     = username,
            filename     = filename,
            percentage   = percentage,
            total        = total,
            plagiarized  = plagiarized,
            trust_verdict= trust["verdict"],
            trust_score  = trust["score"]
        )

        #  PDF Report 
        st.subheader("📥 Download Report")
        pdf_buffer = generate_pdf_report(results, percentage, total, plagiarized)
        st.download_button(
            label    = "⬇️ Download Plagiarism Report (PDF)",
            data     = pdf_buffer,
            file_name= "plagiarism_report.pdf",
            mime     = "application/pdf"
        )