import streamlit as st
import pickle
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Page Configuration & Custom CSS Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Layoff & Hiring Analytics Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border: 1px solid #e9ecef;
        text-align: center;
    }
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        font-size: 1rem;
        color: #64748B;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #2563EB;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #1D4ED8;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Model Loading Function
# -----------------------------------------------------------------------------
@st.cache_resource
def load_model(file_path="randomc.pkl"):
    """Load and cache the pickled scikit-learn Random Forest model safely."""
    try:
        with open(file_path, "rb") as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        return None
    except Exception:
        return None

# Load the model matching your uploaded file name
model = load_model("randomc.pkl")

# Category mappings based on model feature requirements
CATEGORY_MAPS = {
    "company_name": {"Company A": 0, "Company B": 1, "Company C": 2, "Other": 3},
    "industry": {"Tech": 0, "Finance": 1, "Healthcare": 2, "Retail": 3, "Other": 4},
    "country": {"USA": 0, "India": 1, "UK": 2, "Germany": 3, "Other": 4},
    "reason_for_layoffs": {"Cost Cutting": 0, "Restructuring": 1, "AI Automation": 2, "Market Downturn": 3},
    "ai_automation_impact": {"Low": 0, "Medium": 1, "High": 2},
    "ai_replacement_risk": {"Low": 0, "Medium": 1, "High": 2},
    "hiring_trend": {"Decreasing": 0, "Stable": 1, "Increasing": 2},
    "top_hiring_role": {"Software Engineer": 0, "Data Scientist": 1, "Product Manager": 2, "Sales": 3, "Other": 4},
    "ai_adoption_level": {"Low": 0, "Medium": 1, "High": 2}
}


# -----------------------------------------------------------------------------
# Main Application Layout
# -----------------------------------------------------------------------------
def main():
    st.markdown('<div class="main-title">🏢 Workforce Analytics Prediction Portal</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Input company parameters to forecast outcomes using your trained Random Forest model.</div>', unsafe_allow_html=True)

    if model is None:
        st.error("⚠️ Model file (`randomc.pkl`) not found or failed to load. Please ensure it is placed in the root directory.")
        return

    st.markdown("---")

    with st.form("prediction_form"):
        st.subheader("📋 Input Parameters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("##### 🏢 Company Overview")
            company_name_str = st.selectbox("Company Name", list(CATEGORY_MAPS["company_name"].keys()))
            industry_str = st.selectbox("Industry", list(CATEGORY_MAPS["industry"].keys()))
            country_str = st.selectbox("Country", list(CATEGORY_MAPS["country"].keys()))
            company_size = st.number_input("Company Size (Employees)", min_value=1, value=1000, step=50)
            month = st.slider("Month", min_value=1, max_value=12, value=6)
            year = st.number_input("Year", min_value=2000, max_value=2030, value=2026)
            
        with col2:
            st.markdown("##### 📉 Layoffs & AI Exposure")
            layoffs_count = st.number_input("Layoffs Count", min_value=0, value=50, step=10)
            layoff_percentage = st.number_input("Layoff Percentage (%)", min_value=0.0, max_value=100.0, value=5.0, step=0.5)
            reason_str = st.selectbox("Reason for Layoffs", list(CATEGORY_MAPS["reason_for_layoffs"].keys()))
            ai_impact_str = st.selectbox("AI Automation Impact", list(CATEGORY_MAPS["ai_automation_impact"].keys()))
            ai_risk_str = st.selectbox("AI Replacement Risk", list(CATEGORY_MAPS["ai_replacement_risk"].keys()))
            ai_adoption_str = st.selectbox("AI Adoption Level", list(CATEGORY_MAPS["ai_adoption_level"].keys()))

        with col3:
            st.markdown("##### 📈 Hiring & Financial Dynamics")
            open_roles = st.number_input("Open Roles", min_value=0, value=25, step=5)
            hiring_trend_str = st.selectbox("Hiring Trend", list(CATEGORY_MAPS["hiring_trend"].keys()))
            remote_jobs_pct = st.slider("Remote Jobs Percentage (%)", min_value=0.0, max_value=100.0, value=40.0)
            top_role_str = st.selectbox("Top Hiring Role", list(CATEGORY_MAPS["top_hiring_role"].keys()))
            stock_growth = st.number_input("Stock Growth (%)", value=2.5, step=0.5)
            revenue_growth = st.number_input("Revenue Growth (%)", value=5.0, step=0.5)
            salary_budget_change = st.number_input("Salary Budget Change (%)", value=1.0, step=0.5)
            employee_sentiment = st.slider("Employee Sentiment Score", min_value=1.0, max_value=10.0, value=6.5, step=0.1)
            job_security_score = st.slider("Job Security Score", min_value=1.0, max_value=10.0, value=7.0, step=0.1)

        st.markdown("---")
        submit_button = st.form_submit_button(label="🚀 Run Model Prediction")

    if submit_button:
        # Map categorical variables
        input_data = np.array([[
            CATEGORY_MAPS["company_name"][company_name_str],
            CATEGORY_MAPS["industry"][industry_str],
            CATEGORY_MAPS["country"][country_str],
            company_size,
            month,
            year,
            layoffs_count,
            layoff_percentage,
            CATEGORY_MAPS["reason_for_layoffs"][reason_str],
            CATEGORY_MAPS["ai_automation_impact"][ai_impact_str],
            CATEGORY_MAPS["ai_replacement_risk"][ai_risk_str],
            open_roles,
            CATEGORY_MAPS["hiring_trend"][hiring_trend_str],
            remote_jobs_pct,
            CATEGORY_MAPS["top_hiring_role"][top_role_str],
            stock_growth,
            revenue_growth,
            salary_budget_change,
            CATEGORY_MAPS["ai_adoption_level"][ai_adoption_str],
            employee_sentiment,
            job_security_score
        ]])

        try:
            prediction = model.predict(input_data)[0]
            has_proba = hasattr(model, "predict_proba")
            probabilities = model.predict_proba(input_data)[0] if has_proba else None

            st.subheader("🎯 Prediction Output")
            res_col1, res_col2 = st.columns([1, 2])
            
            with res_col1:
                st.markdown(f"""
                    <div class="metric-card">
                        <h4>Predicted Class</h4>
                        <h2 style="color: #2563EB; margin: 0;">Class {prediction}</h2>
                    </div>
                """, unsafe_allow_html=True)

            with res_col2:
                if has_proba:
                    st.write("#### Classification Confidence")
                    proba_df = pd.DataFrame({
                        "Class": [f"Class {c}" for c in model.classes_],
                        "Probability": probabilities
                    })
                    st.bar_chart(proba_df.set_index("Class"))
                else:
                    st.info("Probability estimates are not available for this model configuration.")

        except Exception as err:
            st.error(f"Error during model evaluation: {err}")

if __name__ == "__main__":
    main()