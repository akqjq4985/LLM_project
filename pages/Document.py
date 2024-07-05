import time
from langchain_upstage import (
    UpstageLayoutAnalysisLoader,
    UpstageGroundednessCheck,
    ChatUpstage,
    UpstageEmbeddings,
)
import streamlit as st

st.set_page_config(
    page_title="Document",
    page_icon="📃",
)

def load_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    layzer = UpstageLayoutAnalysisLoader(file_path, output_type="html")
    docs = layzer.load()
    return docs

st.title("Document")

st.markdown(
    """
Welcome!
"""
)

file = st.file_uploader(
    "Upload a .txt .pdf or .docx file",
    type=["pdf", "txt", "docx"],
)

if file:
    docs = load_file(file)
    docs