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
from tavily import TavilyClient
import os 
from requests.exceptions import HTTPError
import json
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
import re
from dotenv import load_dotenv
# from bs4 import BeautifulSoup

load_dotenv()

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

def script_generator(docs):
    # Define prompt
    prompt_template = """
        I need a presentation script for a 20-minute presentation for attached paper. Please ensure that all key points and important details are included so that nothing essential is left out. Make sure to exclude reference section on your presentation script.

        Attached Paper:
        "{text}"

        presentation script format example: 
        #### Slide 1: Title Slide
        "Hello everyone, thank you for joining today. My name is [Your Name], and I'll be presenting on the paper titled 'Denoising Diffusion Probabilistic Models' by Jonathan Ho, Ajay Jain, and Pieter Abbeel from UC Berkeley."

        #### Slide 2: Introduction to Generative Models
        "Generative models have made significant strides in creating high-quality images and audio. Some of the popular models include Generative Adversarial Networks (GANs), Variational Autoencoders (VAEs), autoregressive models, and flow-based models. Each of these models has its strengths and weaknesses. While GANs are known for producing realistic images, they can be challenging to train. VAEs are easier to train but often produce blurry images. Autoregressive models can generate high-fidelity samples but are computationally expensive. Flow-based models ensure exact likelihood but can be complex. This paper introduces a new class of generative models called diffusion probabilistic models, aiming to combine the strengths of these models."

        #### Slide 3: ...

        presentation script for attached paper:
    """
    prompt = PromptTemplate.from_template(prompt_template)
    # Define LLM chain
    llm = ChatUpstage()
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    # Define StuffDocumentsChain
    stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")
    script_data = stuff_chain.run(docs)

    return script_data

def caption_extractor(docs):
    prompt_template = """
        extract captions of all the figures and tables of Attached Paper. You can easily find caption if you focus on the texts right after 'Figure #:' or 'Table #:'. Make sure to never output figure or table that is not in the attached paper.

        Attached Paper:
        "{text}"

        output format example: 
        #### Figure 1:
        "Sampling with classifier-free guidance at high guidance scales using standard methods such as DDPM (Ho et al., 2020) improves image quality but at the cost of diversity, leading to sampled images that look similar in composition. We introduce CADS, a sampling technique that significantly increases the diversity of generations while retaining high output quality."

        #### Figure 2:
        "Low diversity issue in the pose-to-image generation task. (a) The model trained on DeepFashion generates strongly similar outputs. (b) Training on the larger SHHQ dataset only partially solves the issue. (c) Sampling with CADS significantly reduces output similarity."

        #### Table 1: ...

        captions of all the figures and tables of attached paper:
    """
    prompt = PromptTemplate.from_template(prompt_template)
    # Define LLM chain
    llm = ChatUpstage()
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    # Define StuffDocumentsChain
    stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")
    # docs = layzer.load()

    fig_data = stuff_chain.run(docs)

    return fig_data

def caption_slide_mapper(script, captions):
    example = """
        {'output' : 
            {'Silde1' :
                {'Title' : '~'},
                {'Figure' : 
                    {'Requirements' : 'False'},
                    {'Description' : 'None'}
                },
                {'Table' : 
                    {'Requirements' : 'False'},
                    {'Description' : 'None'}
                },
            {'Silde2' :
                {'Title' : '~'},0
                {'Figure' : 
                    {'Requirements' : 'True'},
                    {'Description' : 'Figure # :'}
                },
                {'Table' : 
                    {'Requirements' : 'True'},
                    {'Description' : 'Table # :'}
                },
            ...}
        }
    """
    llm = ChatUpstage(temperature=0.1)

    prompt_template = PromptTemplate.from_template(
        """
        Situation
        - I'm preparing a presentation.
        
        Instructions
        1. Identify the context for each slide.
        2. Match the figure or table in the caption to the corresponding slide and context.
        3. If a slide does not need a figure or table, pass that slide.
        - Take a deep breath and work on this problem step-by-step.
        - I'm going to tip $1000K for a better solution!

        Restrictions
        - It must be content in context.
        - Be sure to follow the Output Format

        Output Format
        - Output format : json
        - Output example :
            "{example}"
        ---
        Script: {Script}
        ---
        Caption : {Caption}
        ---
        """
    )
    chain = prompt_template | llm | StrOutputParser()
    mapping_data = chain.invoke({"example":example,"Script": script, "Caption" : captions})


    return mapping_data

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
    figures_file_path = f"./.cache/files/output-solar.json"
    # loader = UnstructuredFileLoader(file_path, output_type="html")
    f = open(figures_file_path)
    figures = json.load(f)
    # file_2 = st.file_uploader(
    #     "Upload a .txt .pdf or .docx script",
    #     type=["pdf", "txt", "docx"],
    # )
    file_2_path = "./.cache/files/solar_presentation_script.txt"
    loader = UnstructuredFileLoader(file_2_path, output_type="html")
    file_2 = True
    if file_2:
        # script = load_txt(file_2)
        script = script_generator(docs)
        captions = caption_extractor(docs)
        mapping_data = caption_slide_mapper(script, captions)
        script
        ############## script valid check
        # output = extract_small_subject(script)
        # output
        # parser = SlideTitleOutputParser()
        # titles_list = parser.parse(output)
        # titles_list
        ######## ask additional info that has to be searched
        add_info_prompt_template ="""You are an expert in identifing specific keywords or terms in the presentation script that might need further explanation for an audience with no prior knowledge. Suggest 0~2 keywords for additional research that could enhance the presentation.
---
PAPER: {paper}
---
CONTEXT: {context}
---
Restrictions: Return only list of kewords, not other words. The number of keywords should be less than 3. Do not contain direct keywords of the paper.
---
OUTPUT Format: ["keyword", "keyword", ..., "keyword"]
---
OUTPUT: """

        add_info_prompt = PromptTemplate.from_template(add_info_prompt_template)
        chain = add_info_prompt | ChatUpstage(temperature=0.1) | StrOutputParser()

        response = chain.invoke({"history": [], "paper": docs, "context": script})
        response = response.replace('"', '').strip('[]')
        # response
        keywords_list = [keyword.strip() for keyword in response.split(',')]
        keywords_list
        
        ##########################3
        tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
        
        summary_list=[]
        st.markdown("Tavily Results")
        for keyword in keywords_list:
            retries=3
            backoff_factor = 1
            for attempt in range(retries):
                try:
                    response = tavily.search(max_results=3, include_raw_content=True, query=f"what is {keyword}?")
                    response_1 = tavily.search(max_results=1, query=f"what is {keyword}?")
                    response_1
                    break

                except HTTPError as e:
                    if e.response.status_code == 502:
                        if attempt < retries - 1:
                            wait_time = backoff_factor * (2 ** attempt)
                            print(f"Retrying in {wait_time} seconds due to server error...")
                            time.sleep(wait_time)
                        else:
                            raise
                    else:
                        raise
                        

            filter_prompt_template ="""You have expert knowledge in identifying the most relevant content from search results. Your task is to find the most relevant result related to the given keyword in the context. Select and return only one result that best matches the keyword and context.
---
Keyword: {keyword}
---
CONTEXT: {context}
---
Restrictions: Return only the single most relevant result. Do not return more than one result. Ensure the selected result is directly related to the keyword.
---
OUTPUT Format: 
{{
  "title": "Title of the selected result",
  "url": "URL of the selected result",
  "content": "Content of the selected result"
  "raw_content": "Raw content of the selected result"
}}
---
OUTPUT: """

            filter_info_prompt = PromptTemplate.from_template(filter_prompt_template)
            chain = filter_info_prompt | ChatUpstage(temperature=0.1) | StrOutputParser()

            response = chain.invoke({"history": [], "keyword": keyword ,"context": response})
            # response
            summary_prompt_template ="""You are an expert in presenting knowledge from detailed content. Your task is to summarize the given raw content into a 4-5 sentence explanation suitable for a presentation.
---
CONTEXT: {context}
---
Restrictions: Summarize the raw content into 4-5 sentences. Ensure the summary is concise and informative.
---
OUTPUT: 
{{
  "title": "title",
  "url": "url",
  "summary": "Summary of the content"
}}
---
OUTPUT: """
            summary_info_prompt = PromptTemplate.from_template(summary_prompt_template)
            chain = summary_info_prompt | ChatUpstage(temperature=0.1) | StrOutputParser()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            res_splits = text_splitter.split_text(response)
            response = chain.invoke({"history": [], "context": res_splits})
            summary_list.append(response)
        summary_list

        revise_prompt_template = """You are an expert in the field about scripts. Your task is to update the original presentation script by adding new slides based on the provided summaries. Each summary represents a new slide. Place the new slides immediately after the title slide.
---
ORIGINAL SCRIPT: {original_script}
---
NEW SLIDES SUMMARIES: {new_summaries}
---
Restrictions: Ensure that maintain the format of ORIGINAL SCRIPT. Ensure that each new slide has a title with new slide number, the corresponding summary content, and the URL.
---
Instructions: For each summary in the NEW SLIDES SUMMARIES, add a new slide immediately after the title slide in the ORIGINAL SCRIPT. Each new slide should be titled 'Slide #: Additional Background Information', followed by the summary content and the URL in NEW SLIDES SUMMARIES.
---
Example Input for NEW SLIDES SUMMARIES:
[[
    {{
        "title": "JPEG Formats — Progressive vs. Baseline",
        "url": "https://medium.com/hd-pro/jpeg-formats-progressive-vs-baseline-73b3938c2339",
        "summary": "JPEG (Joint Photographic Experts Group) is a widely-used image compression format beneficial for fast downloads and universal browser support. JPEG files use lossy compression, which reduces file size by eliminating some detail, making it suitable for web use. Encoders compress raw images into JPEG, while decoders render the compressed images for display. Cameras and imaging software often include built-in support for JPEG encoding and decoding, streamlining the use of this format in photography and web applications."
    }}
]]
---
Example Output:
#### Slide 1: Title Slide 
Hello everyone, thank you for joining today. My name is [Your Name], and I'll be presenting on the paper titled 'Denoising Diffusion Probabilistic Models' by Jonathan Ho, Ajay Jain, and Pieter Abbeel from UC Berkeley.

#### Slide 2: JPEG Formats — Progressive vs. Baseline
JPEG (Joint Photographic Experts Group) is a widely-used image compression format beneficial for fast downloads and universal browser support. JPEG files use lossy compression, which reduces file size by eliminating some detail, making it suitable for web use. Encoders compress raw images into JPEG, while decoders render the compressed images for display. Cameras and imaging software often include built-in support for JPEG encoding and decoding, streamlining the use of this format in photography and web applications.
URL: https://medium.com/hd-pro/jpeg-formats-progressive-vs-baseline-73b3938c2339

#### Slide 3: Introduction to Generative Models
Generative models have made significant strides in creating high-quality images and audio. Some of the popular models include Generative Adversarial Networks (GANs), Variational Autoencoders (VAEs), autoregressive models, and flow-based models. Each of these models has its strengths and weaknesses. While GANs are known for producing realistic images, they can be challenging to train. VAEs are easier to train but often produce blurry images. Autoregressive models can generate high-fidelity samples but are computationally expensive. Flow-based models ensure exact likelihood but can be complex. This paper introduces a new class of generative models called diffusion probabilistic models, aiming to combine the strengths of these models.

#### Slide 4: Background
Diffusion probabilistic models are a type of latent variable model. They work by using a Markov chain to transform a simple known distribution into a complex data distribution. This transformation process is called the diffusion process, which gradually adds noise to the data. The reverse process then removes this noise to recover the original data distribution. This method is inspired by nonequilibrium thermodynamics.

#### Slide 5: Diffusion Process and Reverse Process
In the diffusion process, noise is added to the data step by step, making the data increasingly noisy until it becomes pure noise. The reverse process aims to undo this, gradually removing the noise to reconstruct the original data. This reverse process is modeled as a Markov chain with learned Gaussian transitions. By training this reverse process using variational inference, we can generate data that closely matches the original data distribution.

#### Slide 6: Training Methodology
The training method used in these models is based on variational inference. This involves optimizing a weighted variational bound, which provides a connection to denoising score matching and Langevin dynamics. This connection simplifies the training process and helps improve the quality of the generated samples.

#### Slide 7: Implementation Details
The model architecture is crucial for achieving high performance. In this implementation, the authors used a U-Net backbone, which is a type of convolutional network known for its effectiveness in image processing tasks. They also incorporated self-attention mechanisms to capture long-range dependencies in the data. The combination of these architectural choices helps the model generate high-quality images.

#### Slide 8: Experimental Setup
The experiments were conducted on well-known datasets like CIFAR10 and LSUN. The evaluation metrics used were the Inception score and FID score, which are standard measures of image quality in generative modeling. These metrics help compare the performance of different models objectively.

---
OUTPUT: The revised presentation script with new slides added immediately after the title slide.
"""
        revise_info_prompt = PromptTemplate.from_template(revise_prompt_template)
        chain = revise_info_prompt | ChatUpstage(temperature=0.1) | StrOutputParser()

        response = chain.invoke({"history": [], "original_script": script, "new_summaries": summary_list})
        response
            
#         ################### summary short sentences
        short_prompt_template ="""You are an expert in summarizing presentation scripts. Please extract the titles of the slides and provide shortened key points for each slide from the following presentation script. Ensure that important keywords are emphasized in your output. Make sure to follow the output format.
    ---
    CONTEXT: {context}
    ---
    OUTPUT FORMAT EXAMPLE:
    #### Slide 1: Title Slide

    - Presentation on 'SOLAR 10.7B' paper by Upstage AI. 
    
    #### Slide 2: Introduction to Large Language Models

    - Large Language Models (LLMs) are ...
    - ...
    ...
    ---
    OUTPUT: """

        short_prompt = PromptTemplate.from_template(short_prompt_template)
        chain = short_prompt | ChatUpstage(temperature=0.1) | StrOutputParser()

        response = chain.invoke({"history": [], "context": response})
        response
        
        
#         ################ make pairs with figures and tables



