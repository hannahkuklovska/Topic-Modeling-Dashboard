import streamlit as st
import pandas as pd

st.title("Topic Modeling App")

#  MODEL SELECT 
model = st.selectbox("Select model", ["LDA", "NMF"])

#  LOAD DATA 
if model == "LDA":
    topics = pd.read_csv("exports_lda/lda_topics_summary.csv")
    users = pd.read_csv("exports_lda/lda_user_topic_distribution.csv")
else:
    topics = pd.read_csv("exports_nmf/nmf_topics_summary.csv")
    users = pd.read_csv("exports_nmf/nmf_user_topic_distribution.csv")

#  TOPICS 
st.header("Topics")

# výber topicu
topic_options = [f"Topic {i+1}" for i in range(len(topics))]
topic_display = st.selectbox("Select topic", topic_options)

# získanie indexu
topic_id = topic_options.index(topic_display)

topic_row = topics.iloc[topic_id]

st.subheader(f"Topic {topic_id + 1}")

st.write("Top tags:")
st.write(topic_row["top_tags"])

st.write("Tag weights:")
st.write(topic_row["top_tag_weights"])

#  USERS 
st.header("User feed")


# labely User1, 2,...
user_labels = ["User " + str(i+1) for i in range(len(users))]
# vyber usera
selected_user = st.selectbox("Select user", user_labels)
#aky ma index
user_index = user_labels.index(selected_user)

#riadok podla indexu
user_row = users.iloc[user_index:user_index+1]

# vyber topic columns
topic_cols = [col for col in users.columns if col.startswith("topic_")]

# hodnoty a labely
values = user_row[topic_cols].iloc[0].values
labels = list(range(1, len(values) + 1))

# dataframe pre graf
chart_data = pd.DataFrame({
    "Topic": labels,
    "Weight": values
}).set_index("Topic")

st.subheader("User topic distribution")
st.bar_chart(chart_data)

# sedi sum to 1? 
user_sum = sum(values)

st.write(f"Sum of topic distribution: {user_sum:.4f}")

if abs(user_sum - 1) < 0.01:
    st.success("Sum is approximately 1")
else:
    st.warning("Sum is not 1")