import pickle
import streamlit as st

# Load models
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Set up page configuration
st.set_page_config(page_title="Review Sentiment Analysis", layout="wide")

# Title and Introduction
st.title("Sentiment Analyzer")
st.markdown("""
Welcome to the  Sentiment Analyzer tool! Please enter a review below to determine if the sentiment is positive or negative.
""")

# Sidebar for input instead of the main page can make the main page less cluttered
with st.sidebar:
    st.header("Enter Your Thought")
    review = st.text_area("Type your review here", height=150)

    # Button in the sidebar
    predict_button = st.button('Predict Sentiment')

# Main page display
if predict_button:
    if review.strip() == "":
        st.error("Please enter a valid text before predicting.")
    else:
        try:
            # Assuming preprocessing is needed for review text
            # This is just an example and needs to be adjusted according to your model's training
            # E.g., you might need to apply TfidfVectorizer or other preprocessing steps
            review_transformed = scaler.transform([review])  # Adjust according to actual preprocessing steps

            # Predict the outcome
            result = model.predict(review_transformed)
            if result[0] == 0:
                st.success('The  sentiment is: **Negative**')
            else:
                st.success('The  sentiment is: **Positive**')
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Optional: Add an about section at the bottom
st.markdown("""
---
**About This Tool**
This tool uses machine learning to analyze the sentiment of text. It's built using Python's Streamlit and scikit-learn library, demonstrating a simple application of natural language processing (NLP) techniques.
""")


