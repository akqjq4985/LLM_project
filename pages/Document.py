import time
from langchain_upstage import (
    UpstageLayoutAnalysisLoader,
    UpstageGroundednessCheck,
    ChatUpstage,
    UpstageEmbeddings,
)
import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts import PromptTemplate
from langchain.document_loaders import UnstructuredFileLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import asyncio
from langchain_community.utils.math import cosine_similarity
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.output_parsers import BaseOutputParser
from langchain.chat_models import ChatOpenAI

st.set_page_config(
    page_title="Document",
    page_icon="📃",
)

class SlideTitleOutputParser(BaseOutputParser):
    def parse(self, text: str) -> list:
        # Remove the brackets and split the text by lines
        lines = text.strip().strip("[]").split('\n')
        
        # Use a regular expression to extract the titles without the indices
        titles = [line.strip('"').split('.')[-1] for line in lines]
        
        return titles
    
class ChromaParallel(Chroma):

    async def afrom_documents(documents, embedding, num_workers=2):
        db = Chroma(embedding_function=embedding)
        # create list of num_workers empty lists
        doc_groups = [[] for _ in range(num_workers)]

        for i in range(len(documents)):
            doc_groups[i % num_workers].append(documents[i])

        tasks = [db.aadd_documents(group) for group in doc_groups]
        await asyncio.gather(*tasks)
        return db

@st.cache_data(show_spinner="Loading paper...") 
def load_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    layzer = UpstageLayoutAnalysisLoader(file_path, output_type="html")
    docs = layzer.load()
    return docs

@st.cache_data(show_spinner="Loading script...") 
def load_txt(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    loader = UnstructuredFileLoader(file_path, output_type="html")
    txt = loader.load()
    return txt

def is_in(question, context):
    is_in_conetxt = """ please determine if the context includes relevent information from the question. 
If the answer for the question is present in the context, please respond with "yes". 
If not, please respond with "no". 
Only provide "yes" or "no" and avoid including any additional information. 
Please do your best. Here is the question and the context:
---
CONTEXT: {context}
---
QUESTION: {question}
---
OUTPUT (yes or no):"""

    is_in_prompt = PromptTemplate.from_template(is_in_conetxt)
    chain = is_in_prompt | ChatUpstage() | StrOutputParser()

    response = chain.invoke({"history": [], "context": context, "question": question})
    return response.lower().startswith("yes")

def retriever_docs(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=UpstageEmbeddings(model="solar-embedding-1-large"))
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    return retriever

def semantic_chunker(
    docs,
    min_chunk_size=100,
    chunk_overlap=10,
    max_chunk_size=1000,
    merge_threshold=0.7,
    embeddings=UpstageEmbeddings(model="solar-embedding-1-large"),
):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=min_chunk_size, chunk_overlap=chunk_overlap
    )
    init_splits = text_splitter.split_documents(docs)
    splits = []

    base_split_text = None
    base_split_emb = None
    for split in init_splits:
        if base_split_text is None:
            base_split_text = split.page_content
            base_split_emb = embeddings.embed_documents([base_split_text])[0]
            continue

        split_emb = embeddings.embed_documents([split.page_content])[0]
        distance = cosine_similarity(X=[base_split_emb], Y=[split_emb])
        if (
            distance[0][0] < merge_threshold
            or len(base_split_text) + len(split.page_content) > max_chunk_size
        ):
            splits.append(Document(page_content=base_split_text))
            base_split_text = split.page_content
            base_split_emb = split_emb
        else:
            base_split_text += split.page_content
    if base_split_text:
        splits.append(Document(page_content=base_split_text))

    return splits

def extract_small_subject(script):
    extract_prompt_template = """ Please extract the titles of the slides from the following presentation script:
---
CONTEXT: {context}
--- 
OUTPUT: """

    extract_prompt = PromptTemplate.from_template(extract_prompt_template)
    chain = extract_prompt | ChatUpstage() | StrOutputParser()

    response = chain.invoke({"history": [], "context": script})
    return response

async def preprocess_docs(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    return splits
    size_vectorstore = FAISS.from_documents(
        documents=splits, embedding=UpstageEmbeddings(model="solar-embedding-1-large")
    )
    size_split_retriever = size_vectorstore.as_retriever(search_kwargs={"k": 3})
    hfembeddings = HuggingFaceEmbeddings(model_name="klue/roberta-small")
    semantic_splits = semantic_chunker(docs,  merge_threshold=0.8, embeddings=hfembeddings)

    semantic_vectorstore = await ChromaParallel.afrom_documents(
        documents=semantic_splits,
        embedding=UpstageEmbeddings(model="solar-embedding-1-large"),
        num_workers=3,
    )
    semantic_split_retriever = semantic_vectorstore.as_retriever(search_kwargs={"k": 6})
    return semantic_split_retriever

async def process_script(file_2, semantic_split_retriever):
    script = load_txt(file_2)
    output = extract_small_subject(script)
    output
    parser = SlideTitleOutputParser()
    titles_list = parser.parse(output)
    titles_list
    for title in titles_list:
        title
        question = f"Do you think you can explain about {title} based only on the CONTEXT?"
        context = semantic_split_retriever.get_relevant_documents(question)
        result = is_in(question, context)
        result

def divide_section(semantic_split_retriever):
    sections = ['Abstract', 'Introduction', 'Method', 'Experiment', 'Conclusion']
    prompt_template = """ please determine if the context includes relevent information from the question. 
If the answer for the question is present in the context, please respond with "yes". 
If not, please respond with "no". 
Only provide "yes" or "no" and avoid including any additional information. 
Please do your best. Here is the question and the context:
---
CONTEXT: {context}
---
QUESTION: {question}
---
OUTPUT (yes or no):"""
    

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
#     semantic_split_retriever = asyncio.run(preprocess_docs(docs))
#     semantic_split_retriever[0]
# #     extract_prompt_template = """You are an expert in explaining academic papers. Your task is to explain the 'Introduction' section from the provided CONTEXT in a way that is easy to understand, just as you would explain it to a friend who is not familiar with the topic. 
# # Please ensure that your explanation is clear, friendly, and covers the main points and purpose of the 'Introduction' section.

# # ---
# # CONTEXT: {context}
# # ---
# # OUTPUT: """ # 잘됨

# #     extract_prompt_template = """You are an expert in extracting specific sections from academic papers. Your task is to extract the entire 'Method' section from the provided CONTEXT. 
# # The 'Method' section starts after the 'Background' or 'Related work' section and ends before the 'Experiment' section. 
# # Please ensure that you return the exact text of the 'Method' section, including any subsections and without adding any additional text or commentary.
# # Think carefully.

# # ---
# # CONTEXT: {context}
# # ---
# # OUTPUT: """ # 잘 안됨
#     section_name = 'Abstract'
#     extract_prompt_template = f"""You are an expert in extracting specific sections from academic papers. Your task is to extract the entire '{section_name}' section from the provided CONTEXT.
#     Give me the '{section_name}' part from CONTEXT. CONTEXT is an academic paper.
#     Please ensure that you return the exact text of the '{section_name}' section, including any subsections and without adding any additional text or commentary.
# ---
# CONTEXT: {{context}}
# ---
# OUTPUT: """

# #     extract_prompt_template = """ You are a good extracter. Give me the list of appeared the number of figures or tables or equations in 'Introduction' part from CONTEXT. CONTEXT is an academic paper.
# # ---
# # CONTEXT: {context}
# # ---
# # OUTPUT: """ # abstract, introduction 어느정도 잘 되는데, equation detect 못함
#     extract_prompt = PromptTemplate.from_template(extract_prompt_template)
#     # chain = extract_prompt | ChatOpenAI(temperature=0.1, model_name="gpt-3.5-turbo-1106") | StrOutputParser()
#     chain = extract_prompt | ChatUpstage() | StrOutputParser()

#     response = chain.invoke({"history": [], "context": semantic_split_retriever[:5]})
#     response
#     # abstract = semantic_split_retriever.get_relevant_documents("Give me the 'Abstract' part from the given academic paper.")
#     # abstract
    file_2 = st.file_uploader(
        "Upload a .txt .pdf or .docx script",
        type=["pdf", "txt", "docx"],
    )

    if file_2:
        script = load_txt(file_2)
        ############## script valid check
        # output = extract_small_subject(script)
        # output
        # parser = SlideTitleOutputParser()
        # titles_list = parser.parse(output)
        # titles_list
        ######## ask additional info that has to be searched
        add_info_prompt_template ="""You are an expert in summarizing presentation scripts and providing additional information for clarity. Your task is to review the provided presentation script and the corresponding academic paper. Identify any parts of the presentation script that might need further explanation for an audience with no prior knowledge. Additionally, suggest 0-2 keywords that would be beneficial to search for and include in the presentation for better understanding.
---
PAPER: {paper}
---
CONTEXT: {context}
---
OUTPUT: """

        add_info_prompt = PromptTemplate.from_template(add_info_prompt_template)
        chain = add_info_prompt | ChatUpstage() | StrOutputParser()

        response = chain.invoke({"history": [], "paper": docs, "context": script})
        response
        ################### summary short sentences
    #     short_prompt_template ="""You are an expert in summarizing presentation scripts. Please extract the titles of the slides and provide shortened key points for each slide from the following presentation script. Ensure that important keywords are emphasized in your output.
    # ---
    # CONTEXT: {context}
    # ---
    # OUTPUT: """

    #     short_prompt = PromptTemplate.from_template(short_prompt_template)
    #     chain = short_prompt | ChatUpstage() | StrOutputParser()

    #     response = chain.invoke({"history": [], "context": script})
    #     response
        
        
        ################ make pairs with figures and tables
        
        # asyncio.run(process_script(file_2, semantic_split_retriever))

