import fitz
import docx
from transformers import pipeline
from langchain.text_splitter import CharacterTextSplitter
import streamlit as st
import torch

# check if CUDA is available and set device accordingly
device = 0 if torch.cuda.is_available() else -1  # 0 for GPU, -1 for CPU
print(f"Device set to {'GPU' if device == 0 else 'CPU'}")

# read text from pdf
def pdfreader(filepath):
    # filepath.seek(0) # reset file pointer
    doc = fitz.open(stream=filepath.read(), filetype = 'pdf')
    text = ''
    for page in doc:
        text += page.get_text()
    return text

# read text from docx
def docreader(filepath):
    # filepath.seek(0) # reset file pointer
    doc = docx.Document(filepath)
    text = []
    for para in doc.paragraphs:
        text.append(para.text)
    return '\n'.join(text)

# chunking text for large documents
chunk_text = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1000,
    chunk_overlap=20
)

# load summarizer model
summarizer = pipeline('summarization', model='facebook/bart-large-cnn', device=device)

# streamlit UI setup
st.title("AI Text Summarizer")
status = st.radio('Upload a PDF/DOCX file or enter text to summarize it.', ('Upload File', 'Enter Text'))
text = ''

if status == 'Upload File':
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx"])
    if uploaded_file is not None:
        if uploaded_file.type == 'application/pdf':
            text = pdfreader(uploaded_file)
            st.write('PDF file uploaded!')
        elif uploaded_file.type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            text = docreader(uploaded_file)
            st.write('DOCX file uploaded!')
        else:
            st.write('File type not supported. Please upload a PDF or DOCX file.')

elif status == 'Enter Text':
    enter_text = st.text_area("Enter your text here:", "Type Here ...")
    if st.button('Submit'):
        text = enter_text
        st.write('Text submitted!')

if text:
    st.text_area("Extracted Text Preview", text[:1000], height=300)
    text = text.replace('\n', ' ') # eradicates newline characters
    try:
        # split the input text into chunks for summarization
        chunks = chunk_text.split_text(text)
        summaries = []
        for chunk in chunks:
            if len(chunk.strip()) > 50: # filter out empty chunks
                summary = summarizer(chunk, max_length=100, min_length=50, do_sample=False)
                summaries.append(summary[0]['summary_text'])

        # join the summaries into output
        final_summary = " ".join(summaries)
        st.write('Here is your summarized text:')
        st.write(final_summary)

    except Exception as e:
        st.error(f"Error occurred: {e}")
else:
    st.warning('No text provided for summarization.')
