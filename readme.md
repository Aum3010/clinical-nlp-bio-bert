# Clinical Decision Support System with Streamlit and Transformers
# WIP

This repository contains a Streamlit web application that leverages the Transformers library for various natural language processing tasks in the medical domain.

## Installation

To run this application, you need to have the following packages installed:

- Python 3.6 or higher
- [Transformers](https://github.com/huggingface/transformers) (version 4.11.3)
- [Torch](https://pytorch.org/get-started/locally/) (version 1.9.0)
- [Tokenizers](https://github.com/huggingface/tokenizers) (version 0.10.3)

You can install these dependencies using the following command:

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Named Entity Recognition (NER)

Identify medical entities in the text using advanced NER techniques. Gain insights into important entities such as diseases, treatments, and more.

### Sentiment Analysis

Analyze the sentiment of medical text to understand the emotional tone and context. Extract valuable insights from patient records, research papers, and clinical notes.

### Question Answering

Get precise answers to medical questions by providing relevant context. Extract vital information from medical documents and research articles with ease.

## Acknowledgments

This application is built upon the foundation of the powerful [dmis-lab/biobert-v1.1](https://huggingface.co/dmis-lab/biobert-v1.1) model from Hugging Face's Transformers library. We are grateful for the open-source contributions that have made this project possible.

## License

This project is licensed under the [MIT License](LICENSE). Feel free to modify and adapt the code to your needs. We encourage contributions and collaboration from the open-source community.
