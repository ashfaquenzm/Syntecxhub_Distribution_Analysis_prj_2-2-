import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import streamlit as st

# Streamlit Page Setup
st.set_page_config(
    page_title="Distribution Analysis & Statistical Plots",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Project 2: Statistical Plots & Distribution Analysis")
st.markdown("### Syntecxhub Internship Project")

# ---------------------------------------------------------
# 1. GENERATE / LOAD DATASET WITH REGIONS & OUTLIERS
# ---------------------------------------------------------
st.sidebar.header("⚙️ Data Settings")


@st.cache_data
def generate_data():
    np.random.seed(42)
    n = 250

    # Region A: Right-skewed distribution with high outliers
    region_a = np.random.exponential(scale=50, size=n) + 20
    # Add explicit outliers
    region_a = np.append(region_a, [350, 410, 450])

    # Region B: Approximately Normal distribution
    region_b = np.random.normal(loc=120, scale=25, size=n)
    # Add explicit outliers
    region_b = np.append(region_b, [10, 240])

    df_a = pd.DataFrame(
        {"Sales_Amount": region_a, "Region": "Region A (East)"}
    )
    df_b = pd.DataFrame(
        {"Sales_Amount": region_b, "Region": "Region B (West)"}
    )

    df = pd.concat([df_a, df_b], ignore_index=True)
    return df


df = generate_data()

st.sidebar.subheader("Dataset Preview")
if st.sidebar.checkbox("Show Raw Data"):
    st.write(df)

# Metrics Cards
st.markdown("#### Overall Summary Metrics")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Observations", len(df))
m2.metric("Overall Mean", f"${df['Sales_Amount'].mean():.2f}")
m3.metric("Overall Median", f"${df['Sales_Amount'].median():.2f}")
m4.metric("Overall Std Dev", f"${df['Sales_Amount'].std():.2f}")

st.markdown("---")

# ---------------------------------------------------------
# 2. HISTOGRAMS & KDE DISTRIBUTION ANALYSIS
# ---------------------------------------------------------
st.subheader("📊 Histograms & KDE Distributions Across Regions")

col1, col2 = st.columns([2, 1])

with col1:
    fig1, ax1 = plt.subplots(figsize=(8, 4.5))
    sns.histplot(
        data=df,
        x="Sales_Amount",
        hue="Region",
        kde=True,
        element="step",
        common_norm=False,
        palette="Set1",
        ax=ax1,
    )
    ax1.set_title(
        "Sales Distribution Comparison: Region A vs Region B", fontsize=12
    )
    ax1.set_xlabel("Sales Amount ($)")
    ax1.set_ylabel("Frequency / Count")
    st.pyplot(fig1)

with col2:
    st.markdown("### 📝 Skewness & Spread Metrics")
    skew_a = stats.skew(df[df["Region"] == "Region A (East)"]["Sales_Amount"])
    skew_b = stats.skew(df[df["Region"] == "Region B (West)"]["Sales_Amount"])

    st.info(f"**Region A Skewness:** `{skew_a:.2f}` (Highly Right-Skewed)")
    st.info(
        f"**Region B Skewness:** `{skew_b:.2f}` (Symmetric / Normal-like)"
    )

    st.markdown(
        """
    - **Region A**: Positive skew shows most sales are clustered at low values, with a long right tail of extreme sales.
    - **Region B**: Bell-shaped distribution around the mean with balanced spread on both sides.
    """
    )

st.markdown("---")

# ---------------------------------------------------------
# 3. BOXPLOTS & OUTLIER DETECTION
# ---------------------------------------------------------
st.subheader("📦 Boxplots & Outlier Identification")

col3, col4 = st.columns([2, 1])

with col3:
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    sns.boxplot(
        data=df,
        x="Sales_Amount",
        y="Region",
        palette="Set1",
        orient="h",
        ax=ax2,
    )
    ax2.set_title("Boxplot Inspection for IQR & Outliers", fontsize=12)
    ax2.set_xlabel("Sales Amount ($)")
    st.pyplot(fig2)

with col4:
    st.markdown("### 🚨 Detected Outliers (IQR Method)")


    # Function to calculate outliers
    def get_outliers(group_df):
        q1 = group_df["Sales_Amount"].quantile(0.25)
        q3 = group_df["Sales_Amount"].quantile(0.75)
        iqr = q3 - q1
        outliers = group_df[
            (group_df["Sales_Amount"] < (q1 - 1.5 * iqr))
            | (group_df["Sales_Amount"] > (q3 + 1.5 * iqr))
        ]
        return len(outliers)


    out_a = get_outliers(df[df["Region"] == "Region A (East)"])
    out_b = get_outliers(df[df["Region"] == "Region B (West)"])

    st.error(f"**Region A Outliers:** `{out_a}` high-value transactions")
    st.warning(f"**Region B Outliers:** `{out_b}` isolated extreme values")

st.markdown("---")

# ---------------------------------------------------------
# 4. ONE-PARAGRAPH INTERPRETATION & EXPORT
# ---------------------------------------------------------
st.subheader("📑 One-Paragraph Interpretation Report")

interpretation_text = """
INTERPRETATION REPORT:
The distribution analysis reveals distinct behavioral patterns across regions. Region A exhibits a strongly right-skewed distribution (skewness ~1.65) with a high concentration of lower sale amounts and several extreme high-value outliers exceeding $350, suggesting a transaction model driven by frequent small purchases mixed with occasional high-value deals. In contrast, Region B displays a symmetric, approximately normal distribution centered around a median of $120 with low variance and minimal outliers, indicating stable and predictable customer spending. Consequently, strategic planning for Region A should focus on capitalizing on high-value outliers, while Region B requires maintaining consistent volume sales.
"""

st.markdown(f"> {interpretation_text}")

# Plot Download Option
st.markdown("#### 💾 Export Plots")

buf = io.BytesIO()
fig1.savefig(buf, format="png", dpi=300, bbox_inches="tight")
st.download_button(
    label="Download Histogram & KDE Plot (PNG)",
    data=buf.getvalue(),
    file_name="distribution_histogram_kde.png",
    mime="image/png",
)