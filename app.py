import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="MediSight AI", page_icon="🏥", layout="wide")

st.markdown("""
<style>
[data-testid="stMetricValue"]{font-size:28px;font-weight:600}
.stTabs [data-baseweb="tab"]{font-size:14px;font-weight:500;padding:10px 20px}
.stTabs [aria-selected="true"]{color:#1D9E75}
</style>
""", unsafe_allow_html=True)

st.title("🏥 MediSight — Clinical AI Assistant")
st.caption("Rajeshwari G D · Clinical Intelligence Platform · MIMIC-III Dataset")

tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Clinical Q&A",
    "🎯 Risk Predictor",
    "📄 Discharge Report",
    "📊 Analytics Dashboard"
])

# ─────────────────────────────────────────────────────────────
# SHARED LOADERS
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_llm():
    from langchain_groq import ChatGroq
    key = os.getenv("GROQ_API_KEY", "")
    for m in ["llama-3.3-70b-versatile",
              "llama3-70b-8192",
              "mixtral-8x7b-32768",
              "gemma2-9b-it"]:
        try:
            llm = ChatGroq(model=m, api_key=key, temperature=0)
            llm.invoke("hi")
            return llm, m
        except Exception:
            continue
    return None, None


@st.cache_resource(show_spinner=False)
def load_rag():
    """
    Load embeddings + vector DB.
    Uses SentenceTransformer directly — avoids ALL langchain-huggingface
    version issues that cause the '_type' error.
    """
    import chromadb
    from sentence_transformers import SentenceTransformer
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    ef = SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    client     = chromadb.PersistentClient(path="./chroma_db")

    # Get the first collection (created by rag_pipeline.py)
    collections = client.list_collections()
    if not collections:
        return None, "No collections found in chroma_db. Run rag_pipeline.py first."

    col_name = collections[0].name
    collection = client.get_collection(name=col_name,
                                       embedding_function=ef)
    return collection, col_name


def search_docs(collection, query, n=4):
    """Query chromadb directly — returns list of text chunks."""
    results = collection.query(
        query_texts=[query],
        n_results=min(n, collection.count())
    )
    docs = results["documents"][0] if results["documents"] else []
    meta = results["metadatas"][0] if results["metadatas"] else []
    return docs, meta


# ══════════════════════════════════════════════════════════════
# TAB 1 — Clinical Q&A
# ══════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Ask questions about discharge summaries")

    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        st.error("GROQ_API_KEY not found. Add it to your .env file.")
        st.code("GROQ_API_KEY=gsk_your_key_here", language="bash")
        st.stop()

    if not os.path.exists("./chroma_db"):
        st.error("chroma_db not found. Run: python rag_pipeline.py")
        st.stop()

    with st.spinner("Loading AI components..."):
        try:
            llm, llm_name = load_llm()
            if llm is None:
                st.error("No Groq model available. Check your API key at console.groq.com")
                st.stop()

            collection, col_info = load_rag()
            if collection is None:
                st.error(col_info)
                st.stop()

            st.success("AI ready — Model: " + llm_name +
                       " | Docs indexed: " + str(collection.count()))
        except Exception as e:
            st.error("Error loading AI: " + str(e))
            st.stop()

    # Example buttons
    st.markdown("**Try an example:**")
    examples = [
        "Which patients were readmitted within 30 days?",
        "Which diabetic patients were readmitted?",
        "What medications were given to heart failure patients?",
        "Which patients were discharged to SNF?",
        "Summarize all COPD cases",
        "Which patients had sepsis?"
    ]
    cols = st.columns(3)
    for i, ex in enumerate(examples):
        if cols[i % 3].button(ex, key="ex" + str(i)):
            st.session_state["q"] = ex

    query = st.text_input(
        "Your question:",
        placeholder="Which diabetic patients were readmitted?",
        value=st.session_state.get("q", "")
    )

    if query:
        with st.spinner("Searching discharge summaries..."):
            try:
                docs, meta = search_docs(collection, query, n=4)

                if not docs:
                    st.warning("No matching documents found.")
                else:
                    context  = "\n\n".join(docs)
                    prompt   = (
                        "You are a clinical AI assistant. Use ONLY the discharge "
                        "notes below to answer the question. Be specific.\n\n"
                        "DISCHARGE NOTES:\n" + context +
                        "\n\nQUESTION: " + query +
                        "\n\nANSWER:"
                    )
                    response = llm.invoke(prompt)
                    answer   = (response.content
                                if hasattr(response, "content")
                                else str(response))

                    st.markdown("### AI Answer")
                    st.info(answer)

                    with st.expander("Source documents used"):
                        for i, (doc, m) in enumerate(zip(docs, meta), 1):
                            src = os.path.basename(m.get("source", "Document " + str(i)))
                            st.markdown("**Source " + str(i) + ":** `" + src + "`")
                            st.caption(doc[:300] + "...")

            except Exception as e:
                st.error("Error: " + str(e))

# ══════════════════════════════════════════════════════════════
# TAB 2 — Risk Predictor
# ══════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Predict 30-day readmission risk")
    st.markdown("Adjust sliders then click Predict.")

    c1, c2 = st.columns(2)
    with c1:
        age   = st.slider("Patient Age",            18, 90, 65)
        los   = st.slider("Length of Stay (days)",   1, 30,  7)
        nd    = st.slider("Number of Diagnoses",      1, 20,  8)
        drugs = st.slider("Number of Drugs",          1, 50, 15)
    with c2:
        gender = st.selectbox("Gender",   ["Male","Female"])
        ins    = st.selectbox("Insurance", ["Medicare","Medicaid",
                                            "Private","Government","Self Pay"])
        dc     = st.selectbox("Discharge Destination",
                              ["HOME","HOME HEALTH CARE","SNF",
                               "REHAB/DISTINCT PART HOSP","HOSPICE-HOME"])

    if st.button("Predict Readmission Risk", type="primary"):
        model = None
        for mp in ["outputs/readmission_model.pkl",
                   "../MediSight_ML/outputs/readmission_model.pkl",
                   "readmission_model.pkl"]:
            if os.path.exists(mp):
                model = joblib.load(mp)
                break

        if model is None:
            st.error("readmission_model.pkl not found. Run ML notebook first.")
        else:
            g_enc = 0 if gender == "Female" else 1
            i_enc = {"Government":0,"Medicaid":1,"Medicare":2,
                     "Private":3,"Self Pay":4}.get(ins, 2)
            d_enc = {"HOME":0,"HOME HEALTH CARE":1,
                     "HOME WITH HOME IV PROVIDR":2,"HOSPICE-HOME":3,
                     "LONG TERM CARE HOSPITAL":4,
                     "REHAB/DISTINCT PART HOSP":5,"SNF":6}.get(dc, 0)
            hc = int(nd >= 9 and drugs >= 20)
            el = int(age >= 75)
            ls = int(los >= 7)

            try:
                risk = float(model.predict_proba(
                    [[age,g_enc,los,i_enc,d_enc,0,nd,drugs,hc,el,ls]]
                )[0][1])
            except Exception:
                try:
                    risk = float(model.predict_proba(
                        [[age,g_enc,los,i_enc,d_enc,nd,drugs]]
                    )[0][1])
                except Exception as e2:
                    st.error("Prediction error: " + str(e2))
                    st.stop()

            tier = ("HIGH RISK"   if risk >= 0.60 else
                    "MEDIUM RISK" if risk >= 0.30 else "LOW RISK")

            st.markdown("---")
            m1, m2, m3 = st.columns(3)
            m1.metric("Risk Score", str(round(risk, 3)))
            m2.metric("Risk %",     str(round(risk*100, 1)) + "%")
            m3.metric("Risk Tier",  tier)
            st.progress(float(risk))

            if   risk >= 0.60: st.error("HIGH RISK — Intensive discharge planning + 7-day call.")
            elif risk >= 0.30: st.warning("MEDIUM RISK — 14-day follow-up recommended.")
            else:              st.success("LOW RISK — Standard discharge. 30-day follow-up.")

            st.markdown("### Key risk factors:")
            factors = []
            if age >= 75:  factors.append("Age " + str(age) + " — elderly higher readmission risk")
            if los >= 7:   factors.append("LOS " + str(los) + " days — strongest predictor")
            if ins in ["Medicare","Medicaid"]: factors.append(ins + " — higher risk population")
            if dc in ["SNF","REHAB/DISTINCT PART HOSP"]: factors.append("Discharged to " + dc)
            if nd >= 9:    factors.append(str(nd) + " diagnoses — complex patient")
            if drugs >= 20:factors.append(str(drugs) + " drugs — high medication burden")
            if not factors:factors.append("No major risk flags — low risk profile")
            for f in factors:
                st.markdown("- " + f)

# ══════════════════════════════════════════════════════════════
# TAB 3 — Discharge Report Generator
# ══════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Auto Discharge Report Generator")
    st.markdown("Fill patient details — AI writes the discharge summary — download as PDF.")

    with st.form("report_form"):
        st.markdown("#### Patient Information")
        f1, f2, f3 = st.columns(3)
        with f1:
            r_name   = st.text_input("Patient Name",   "John D")
            r_age    = st.number_input("Age", 1, 110, 65)
            r_gender = st.selectbox("Gender_r", ["Male","Female"],
                                    label_visibility="collapsed")
            st.caption("Gender")
        with f2:
            r_id     = st.text_input("Patient ID",     "P-10035")
            r_ins    = st.selectbox("Insurance_r",
                                    ["Medicare","Medicaid","Private",
                                     "Government","Self Pay"],
                                    label_visibility="collapsed")
            st.caption("Insurance")
            r_los    = st.number_input("LOS (days)", 1, 90, 7)
        with f3:
            r_admit  = st.text_input("Admit Date",     "2024-01-15")
            r_disch  = st.text_input("Discharge Date", "2024-01-22")
            r_dc_loc = st.selectbox("Discharge To_r",
                                    ["HOME","HOME HEALTH CARE","SNF",
                                     "REHAB/DISTINCT PART HOSP","HOSPICE-HOME"],
                                    label_visibility="collapsed")
            st.caption("Discharge To")

        st.markdown("#### Clinical Details")
        g1, g2 = st.columns(2)
        with g1:
            r_diag  = st.text_area("Primary Diagnosis",
                                   "Acute Decompensated Heart Failure", height=70)
            r_comor = st.text_area("Comorbidities",
                                   "Hypertension, Type 2 Diabetes, CKD Stage 3", height=70)
            r_meds  = st.text_area("Discharge Medications",
                                   "Furosemide 40mg, Carvedilol 6.25mg, Metformin 1000mg",
                                   height=70)
        with g2:
            r_proc  = st.text_area("Procedures Performed",
                                   "Echocardiogram, BNP test, IV Furosemide", height=70)
            r_vital = st.text_area("Discharge Vitals",
                                   "BP 128/76, HR 72, O2 Sat 97%, Temp 98.6F", height=70)
            r_fu    = st.text_area("Follow-up Instructions",
                                   "Cardiology in 1 week, daily weights, low sodium diet",
                                   height=70)

        r_risk = st.select_slider("Readmission Risk",
                                  ["Low Risk","Medium Risk","High Risk"],
                                  "Medium Risk")

        do_report = st.form_submit_button(
            "Generate Discharge Report", type="primary",
            use_container_width=True
        )

    if do_report:
        api_key = os.getenv("GROQ_API_KEY","")
        if not api_key:
            st.error("GROQ_API_KEY not found.")
            st.stop()

        prompt = (
            "You are a senior hospital physician. Write a professional "
            "clinical discharge summary. Use formal medical language.\n\n"
            "Patient: " + r_name + " | ID: " + r_id +
            " | Age: " + str(r_age) + " | Gender: " + r_gender +
            " | Insurance: " + r_ins + "\n"
            "Admitted: " + r_admit + " | Discharged: " + r_disch +
            " | LOS: " + str(r_los) + " days | Discharged to: " + r_dc_loc + "\n"
            "Diagnosis: " + r_diag + "\n"
            "Comorbidities: " + r_comor + "\n"
            "Procedures: " + r_proc + "\n"
            "Medications: " + r_meds + "\n"
            "Vitals: " + r_vital + "\n"
            "Follow-up: " + r_fu + "\n"
            "Risk: " + r_risk + "\n\n"
            "Sections:\n"
            "1. PATIENT INFORMATION\n"
            "2. REASON FOR ADMISSION\n"
            "3. HOSPITAL COURSE\n"
            "4. PROCEDURES AND INVESTIGATIONS\n"
            "5. DISCHARGE CONDITION AND VITALS\n"
            "6. DISCHARGE MEDICATIONS\n"
            "7. FOLLOW-UP INSTRUCTIONS\n"
            "8. READMISSION RISK ASSESSMENT\n"
            "9. PHYSICIAN SIGNATURE\n\n"
            "400-500 words. Be specific and clinical."
        )

        with st.spinner("AI is writing the discharge summary..."):
            try:
                rep_llm, _ = load_llm()
                if rep_llm is None:
                    st.error("Groq LLM unavailable.")
                    st.stop()

                resp       = rep_llm.invoke(prompt)
                report_txt = resp.content if hasattr(resp,"content") else str(resp)

                st.markdown("---")
                st.markdown("### Generated Discharge Summary")
                st.markdown(
                    "<div style='background:#f8f9fa;border-left:4px solid #1D9E75;"
                    "padding:20px;border-radius:0 8px 8px 0;font-size:13px;"
                    "line-height:1.8;white-space:pre-wrap'>"
                    + report_txt.replace("<","&lt;").replace(">","&gt;")
                    + "</div>",
                    unsafe_allow_html=True
                )

                # PDF
                try:
                    from fpdf import FPDF
                    from fpdf.enums import XPos, YPos

                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_margins(15,15,15)

                    pdf.set_fill_color(8,80,65)
                    pdf.set_text_color(255,255,255)
                    pdf.set_font("Helvetica","B",14)
                    pdf.cell(0,12,"MediSight Hospital Discharge Summary",
                             new_x=XPos.LMARGIN, new_y=YPos.NEXT,
                             fill=True, align="C")

                    pdf.set_text_color(0,0,0)
                    pdf.set_fill_color(230,245,238)
                    pdf.set_font("Helvetica","",10)
                    meta_str = (r_name + "  |  ID: " + r_id +
                                "  |  Age: " + str(r_age) +
                                "  |  Admitted: " + r_admit +
                                "  |  Discharged: " + r_disch)
                    pdf.cell(0,8,meta_str,
                             new_x=XPos.LMARGIN,new_y=YPos.NEXT,
                             fill=True,align="C")
                    pdf.ln(4)

                    safe = report_txt.encode("latin-1",errors="replace").decode("latin-1")
                    for line in safe.split("\n"):
                        line = line.strip()
                        if not line:
                            pdf.ln(3)
                            continue
                        if line.isupper() or (line.endswith(":") and len(line)<50):
                            pdf.set_font("Helvetica","B",10)
                            pdf.set_fill_color(240,240,240)
                            pdf.cell(0,7,line,
                                     new_x=XPos.LMARGIN,new_y=YPos.NEXT,fill=True)
                            pdf.set_font("Helvetica","",10)
                        else:
                            pdf.set_font("Helvetica","",10)
                            pdf.multi_cell(0,6,line)

                    pdf.ln(4)
                    bc = (200,40,40) if "High" in r_risk else \
                         (180,120,0) if "Medium" in r_risk else (8,80,65)
                    pdf.set_fill_color(*bc)
                    pdf.set_text_color(255,255,255)
                    pdf.set_font("Helvetica","B",10)
                    pdf.cell(0,8,"Risk: " + r_risk + "  |  MediSight AI",
                             new_x=XPos.LMARGIN,new_y=YPos.NEXT,
                             fill=True,align="C")

                    st.download_button(
                        "Download Report as PDF",
                        data=bytes(pdf.output()),
                        file_name="discharge_" + r_id.replace("-","_") + ".pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )

                except Exception as pdf_err:
                    st.warning("PDF failed (" + str(pdf_err) + ") — downloading as TXT.")
                    st.download_button(
                        "Download Report as TXT",
                        data=report_txt.encode("utf-8"),
                        file_name="discharge_" + r_id.replace("-","_") + ".txt",
                        mime="text/plain",
                        use_container_width=True
                    )

            except Exception as e:
                st.error("Report generation failed: " + str(e))

# ══════════════════════════════════════════════════════════════
# TAB 4 — Analytics Dashboard
# ══════════════════════════════════════════════════════════════
 
with tab4:
    st.subheader("Live Analytics Dashboard")
    st.markdown("Interactive charts powered by your ML model outputs.")

    rdf = None
    for rp in ["outputs/risk_scores.csv",
               "../MediSight_ML/outputs/risk_scores.csv",
               "risk_scores.csv"]:
        if os.path.exists(rp):
            rdf = pd.read_csv(rp, encoding="latin-1")
            break

    if rdf is None:
        st.warning("risk_scores.csv not found. Run ML notebook Cell 13 first.")
        st.stop()

    if "risk_tier" not in rdf.columns:
        rdf["risk_tier"] = rdf["risk_score"].apply(
            lambda x: "High Risk" if x>=0.60 else
                      "Medium Risk" if x>=0.30 else "Low Risk"
        )
    if "risk_pct" not in rdf.columns:
        rdf["risk_pct"] = (rdf["risk_score"]*100).round(1)

    try:
        import plotly.express as px
        COLORS = {"High Risk":"#D85A30","Medium Risk":"#EF9F27","Low Risk":"#1D9E75"}

        # KPIs
        total       = len(rdf)
        readmitted  = int(rdf["target"].sum()) if "target" in rdf.columns else 0
        high_risk   = int((rdf["risk_tier"]=="High Risk").sum())
        avg_risk    = round(float(rdf["risk_score"].mean()),3)
        readmit_pct = round(readmitted/total*100,1) if total else 0

        k1,k2,k3,k4 = st.columns(4)
        k1.metric("Total Patients",    str(total))
        k2.metric("Readmitted (30d)",  str(readmitted)+" ("+str(readmit_pct)+"%)")
        k3.metric("High Risk Patients",str(high_risk))
        k4.metric("Avg Risk Score",    str(avg_risk))
        st.markdown("---")

        # Row 1
        rc1, rc2 = st.columns(2)
        with rc1:
            td = (rdf["risk_tier"].value_counts()
                      .reindex(["High Risk","Medium Risk","Low Risk"])
                      .reset_index())
            td.columns = ["Risk Tier","Count"]
            fig1 = px.bar(td,x="Risk Tier",y="Count",
                          color="Risk Tier",color_discrete_map=COLORS,
                          text="Count",title="Patient Count by Risk Tier")
            fig1.update_traces(textposition="outside",marker_line_width=0)
            fig1.update_layout(showlegend=False,height=320,
                               plot_bgcolor="rgba(0,0,0,0)",
                               paper_bgcolor="rgba(0,0,0,0)",
                               margin=dict(t=40,b=20,l=20,r=20))
            st.plotly_chart(fig1,use_container_width=True)

        with rc2:
            fig2 = px.pie(td,names="Risk Tier",values="Count",
                          color="Risk Tier",color_discrete_map=COLORS,
                          title="Risk Tier Split",hole=0.55)
            fig2.update_traces(textposition="outside",textinfo="percent+label",
                               marker=dict(line=dict(color="white",width=2)))
            fig2.update_layout(height=320,showlegend=False,
                               paper_bgcolor="rgba(0,0,0,0)",
                               margin=dict(t=40,b=20,l=20,r=20))
            st.plotly_chart(fig2,use_container_width=True)

        # Row 2
        rc3, rc4 = st.columns(2)
        with rc3:
            fig3 = px.histogram(rdf,x="risk_pct",nbins=10,
                                color_discrete_sequence=["#1D9E75"],
                                title="Risk Score Distribution",
                                labels={"risk_pct":"Risk Score (%)"})
            fig3.update_traces(marker_line_color="white",marker_line_width=1)
            fig3.update_layout(height=300,
                               plot_bgcolor="rgba(0,0,0,0)",
                               paper_bgcolor="rgba(0,0,0,0)",
                               margin=dict(t=40,b=20,l=20,r=20))
            st.plotly_chart(fig3,use_container_width=True)

        with rc4:
            if "insurance" in rdf.columns:
                id2 = (rdf.groupby("insurance")["risk_score"]
                           .mean().reset_index()
                           .sort_values("risk_score"))
                id2.columns = ["Insurance","Avg Risk"]
                id2["Color"] = id2["Avg Risk"].apply(
                    lambda x: "#D85A30" if x>=0.6 else
                              "#EF9F27" if x>=0.3 else "#1D9E75"
                )
                fig4 = px.bar(id2,x="Avg Risk",y="Insurance",
                              orientation="h",color="Color",
                              color_discrete_map="identity",
                              text=id2["Avg Risk"].round(3),
                              title="Avg Risk Score by Insurance")
                fig4.update_traces(textposition="outside",marker_line_width=0)
                fig4.update_layout(showlegend=False,height=300,
                                   plot_bgcolor="rgba(0,0,0,0)",
                                   paper_bgcolor="rgba(0,0,0,0)",
                                   margin=dict(t=40,b=20,l=20,r=20),
                                   xaxis=dict(range=[0,1]))
                st.plotly_chart(fig4,use_container_width=True)
            else:
                st.info("Add insurance column to risk_scores.csv for this chart.")

        st.write("Columns in risk_scores.csv:", list(rdf.columns))        

        # Top 20 table
        st.markdown("### Top 20 Highest-Risk Patients")

        # Use only columns that actually exist in the CSV
        all_possible = ["subject_id","age","insurance","los_days",
                        "risk_score","risk_pct","risk_tier","target","prediction"]
        show = [c for c in all_possible if c in rdf.columns]

        # If no identifier column exists, add row number as Patient ID
        if "subject_id" not in rdf.columns:
            rdf["Patient #"] = ["P-" + str(10000+i) for i in range(len(rdf))]
            show = ["Patient #"] + show

        top20 = (rdf.sort_values("risk_score", ascending=False)
                    .head(20)[show].reset_index(drop=True))
        top20.index += 1

        

        def color_tier(val):
            if val=="High Risk":   return "background-color:#FAECE7;color:#993C1D;font-weight:600"
            if val=="Medium Risk": return "background-color:#FAEEDA;color:#854F0B;font-weight:500"
            return "background-color:#E1F5EE;color:#085041"

        def color_score(val):
            try:
                v=float(val)
                if v>=0.60: return "color:#993C1D;font-weight:600"
                if v>=0.30: return "color:#854F0B;font-weight:500"
                return "color:#085041"
            except Exception:
                return ""

        # map() replaced applymap() in pandas 2.1+
        try:
            styled = (top20.style
                           .map(color_tier,  subset=["risk_tier"])
                           .map(color_score, subset=["risk_score"]))
        except AttributeError:
            # Fallback for older pandas
            styled = (top20.style
                           .applymap(color_tier,  subset=["risk_tier"])
                           .applymap(color_score, subset=["risk_score"]))

        st.dataframe(styled, use_container_width=True, height=480)
        st.download_button("Download Top 20 as CSV",
                           data=top20.to_csv(index=False).encode("utf-8"),
                           file_name="top20_high_risk.csv",
                           mime="text/csv")

    except ImportError:
        st.error("Plotly not installed. Run: pip install plotly")