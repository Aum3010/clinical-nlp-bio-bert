import streamlit as st
from transformers import pipeline

# Load the BioBERT NER model
ner_model = pipeline("ner", model="dmis-lab/biobert-v1.1", tokenizer="dmis-lab/biobert-v1.1")

sentiment_model = pipeline("sentiment-analysis", model="dmis-lab/biobert-v1.1", tokenizer="dmis-lab/biobert-v1.1")

qa_model = pipeline("question-answering", model="dmis-lab/biobert-v1.1", tokenizer="dmis-lab/biobert-v1.1")

summarization_model = pipeline("summarization", model="dmis-lab/biobert-v1.1", tokenizer="dmis-lab/biobert-v1.1")

# Define the Streamlit app
def main():
    st.title("Medical Diagnosis and Patient Care")
    st.subheader("Clinical Decision Support System")
    input_text = st.text_area("Enter Medical Text:", height=200)
    functionality = st.selectbox("Select Functionality:", ["Named Entity Recognition", "Sentiment Analysis", "Question Answering"])
    if st.button("Submit"):
        if functionality == "Named Entity Recognition":
            perform_ner(input_text)
        elif functionality == "Sentiment Analysis":
            perform_sentiment_analysis(input_text)
        elif functionality == "Question Answering":
            perform_question_answering(input_text)
        elif functionality == "Text Summarization":
            perform_text_summarization(input_text)

def perform_ner(text):
    st.subheader("Named Entity Recognition (NER) Results")
    entities = ner_model(text)
    print(entities)
    for entity in entities:
        st.write(f"Entity: {entity['entity']}, Confidence: {entity['score']}")

def perform_sentiment_analysis(text):
    st.subheader("Sentiment Analysis Results")
    sentiment = sentiment_model(text)
    st.write(f"Sentiment: {sentiment[0]['label']}, Confidence: {sentiment[0]['score']}")

def perform_question_answering(question):
    st.subheader("Question Answering Results")
    context = st.text_area("Enter Context for Question Answering:", height=100)
    qa_input = {"question": question, "context": context}
    answer = qa_model(qa_input)
    st.write(f"Question: {question}")
    st.write(f"Answer: {answer['answer']}, Confidence: {answer['score']}")

# Function to perform Text Summarization
def perform_text_summarization(text):
    st.subheader("Text Summarization Results")
    summary = summarization_model(text, max_length=100, min_length=30, do_sample=False)[0]
    st.write(f"Summary: {summary['summary_text']}")

# Run the app
if __name__ == "__main__":
    main()
