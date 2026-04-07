import streamlit as st
import pandas as pd

st.title("Topic Modeling App")

# ===== MODEL SELECT =====
model = st.selectbox("Select model", ["LDA", "NMF"])

# ===== LOAD DATA =====
if model == "LDA":
    topics = pd.read_csv("exports_lda/lda_topics_summary.csv")
    users = pd.read_csv("exports_lda/lda_user_topic_distribution.csv")
else:
    topics = pd.read_csv("exports_nmf/nmf_topics_summary.csv")
    users = pd.read_csv("exports_nmf/nmf_user_topic_distribution.csv")

# ===== TOPICS =====
st.header("Topics")

# zobrazovanie od 1
topic_display = st.selectbox(
    "Select topic",
    [f"Topic {i+1}" for i in topics["topic_id"]]
)

# späť na index (0-based)
topic_id = int(topic_display.split(" ")[1]) - 1

topic_row = topics[topics["topic_id"] == topic_id].iloc[0]

st.subheader(f"Topic {topic_id + 1}")

st.write("Top tags:")
st.write(topic_row["top_tags"])

st.write("Tag weights:")
st.write(topic_row["top_tag_weights"])

# ===== USERS =====
st.header("User feed")

user_id = st.selectbox("Select user", users["user_email"])
user_row = users[users["user_email"] == user_id]

topic_cols = [col for col in users.columns if col.startswith("topic_")]

# zoradenie podľa váhy
chart_df = user_row[topic_cols].T

# premenovanie indexu: topic_0 → Topic 1
chart_df.index = [f"Topic {int(col.split('_')[1]) + 1}" for col in chart_df.index]

# zoradenie
chart_df = chart_df.sort_values(by=user_row.index[0], ascending=False)

st.bar_chart(chart_df)

# sum of topic weights
user_sum = user_row[topic_cols].sum(axis=1).iloc[0]

st.write(f"Sum of topic distribution: {user_sum:.4f}")

if abs(user_sum - 1) < 0.01:
    st.success(f"Sum ≈ 1 ({user_sum:.4f})")
else:
    st.warning(f"Sum not 1 ({user_sum:.4f})")