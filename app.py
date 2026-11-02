
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Zero Trust Decision Support System",
    page_icon="🛡️",
    layout="wide"
)

# --------------------------------------------------
# LOAD RESEARCH DATA
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

@st.cache_data
def load_data():
    matrix = pd.read_csv(
        BASE_DIR / "zero_trust_decision_matrix.csv",
        index_col="Alternative"
    )

    weights = pd.read_csv(
        BASE_DIR / "zero_trust_scenario_weights.csv",
        index_col="Criterion"
    )

    return matrix, weights


decision_matrix, scenario_weights = load_data()

# Ensure criteria appear in the correct order
scenario_weights = scenario_weights.loc[
    decision_matrix.columns
]

# --------------------------------------------------
# TOPSIS CALCULATION
# --------------------------------------------------

def calculate_topsis(matrix, weights):
    weights = np.asarray(weights, dtype=float)

    assert len(weights) == matrix.shape[1]
    assert np.isclose(weights.sum(), 1.0)
    assert np.all(weights > 0)

    normalized = matrix.div(
        np.sqrt((matrix ** 2).sum(axis=0)),
        axis=1
    )

    weighted = normalized.mul(weights, axis=1)

    positive_ideal = weighted.max(axis=0)
    negative_ideal = weighted.min(axis=0)

    d_positive = np.sqrt(
        ((weighted - positive_ideal) ** 2).sum(axis=1)
    )

    d_negative = np.sqrt(
        ((weighted - negative_ideal) ** 2).sum(axis=1)
    )

    closeness = d_negative / (d_positive + d_negative)

    results = pd.DataFrame({
        "Closeness Coefficient": closeness
    })

    results["Rank"] = (
        results["Closeness Coefficient"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    return results.sort_values("Rank")


# --------------------------------------------------
# APPLICATION HEADER
# --------------------------------------------------

st.title("🛡️ Zero Trust Decision Support System")

st.write(
    "An interactive framework for selecting Zero Trust "
    "implementation approaches based on organizational "
    "security requirements and preferences."
)

st.caption(
    "Research prototype using hierarchical AHP and TOPSIS."
)

st.divider()

# --------------------------------------------------
# ORGANIZATIONAL SCENARIO SELECTION
# --------------------------------------------------

st.subheader("1. Select Organizational Scenario")

selected_scenario = st.selectbox(
    "Choose an organization type:",
    options=scenario_weights.columns.tolist()
)

selected_weights = scenario_weights[
    selected_scenario
]

# --------------------------------------------------
# CALCULATE RECOMMENDATION
# --------------------------------------------------

ranking = calculate_topsis(
    decision_matrix,
    selected_weights.to_numpy()
)

winner = ranking.index[0]
winner_score = ranking.iloc[0][
    "Closeness Coefficient"
]

# --------------------------------------------------
# DISPLAY RECOMMENDATION
# --------------------------------------------------

st.subheader("2. Recommended Zero Trust Approach")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Highest-Ranked Approach",
        winner
    )

with col2:
    st.metric(
        "TOPSIS Closeness Coefficient",
        f"{winner_score:.4f}"
    )

st.info(
    "This recommendation reflects the selected hypothetical "
    "scenario and provisional suitability ratings. "
    "It is not a validated deployment recommendation."
)

# --------------------------------------------------
# TOPSIS RANKING
# --------------------------------------------------

st.subheader("3. TOPSIS Ranking Results")

st.dataframe(
    ranking.style.format({
        "Closeness Coefficient": "{:.4f}"
    }),
    use_container_width=True
)

st.subheader("4. Comparison of Zero Trust Approaches")

short_names = {
    "Enhanced Identity Governance (EIG)": "EIG",
    "Software-Defined Perimeter (SDP)": "SDP",
    "Microsegmentation": "Microsegmentation",
    "Secure Access Service Edge (SASE)": "SASE"
}

chart_data = ranking.rename(index=short_names)

st.bar_chart(
    chart_data["Closeness Coefficient"],
    horizontal=True,
    x_label="Zero Trust Approach",
    y_label="TOPSIS Closeness Coefficient"
)

# --------------------------------------------------
# AHP CRITERION WEIGHTS
# --------------------------------------------------

st.subheader("5. Organizational Criterion Weights")

criterion_names = {
    "C1": "Infrastructure Environment",
    "C2": "Legacy-System Integration",
    "C3": "Scalability",
    "C4": "Performance",
    "C5": "Cost (Affordability)",
    "C6": "Implementation Complexity",
    "C7": "Security Requirements",
    "C8": "Identity and Access Requirements",
    "C9": "Remote and Distributed Access"
}

weights_table = pd.DataFrame({
    "Criterion": selected_weights.index,
    "Description": [
        criterion_names[code]
        for code in selected_weights.index
    ],
    "Global Weight": selected_weights.values,
    "Weight (%)": selected_weights.values * 100
})

weights_table = weights_table.sort_values(
    "Global Weight",
    ascending=False
)

st.dataframe(
    weights_table.style.format({
        "Global Weight": "{:.4f}",
        "Weight (%)": "{:.2f}"
    }),
    hide_index=True,
    use_container_width=True
)

# --------------------------------------------------
# METHODOLOGY
# --------------------------------------------------

with st.expander("About the AHP-TOPSIS Methodology"):
    st.markdown("""
    **Analytic Hierarchy Process (AHP)** determines
    the relative importance of organizational criteria.

    **Technique for Order Preference by Similarity
    to Ideal Solution (TOPSIS)** ranks alternatives
    according to their distances from the positive
    and negative ideal solutions.

    The prototype evaluates four approaches:

    - Enhanced Identity Governance (EIG)
    - Software-Defined Perimeter (SDP)
    - Microsegmentation
    - Secure Access Service Edge (SASE)

    The evaluation uses nine criteria grouped into
    Technical, Business, and Security factors.

    All criteria are benefit-coded, meaning higher
    suitability ratings are preferable.
    """)

st.divider()

st.caption(
    "Academic research prototype | "
    "AHP-TOPSIS Zero Trust Selection Framework"
)
