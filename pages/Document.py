import time
from langchain_upstage import (
    UpstageLayoutAnalysisLoader,
    UpstageGroundednessCheck,
    ChatUpstage,
    UpstageEmbeddings,
)
import streamlit as st
from bs4 import BeautifulSoup
import os
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_core.output_parsers import StrOutputParser
import re
import os
from dotenv import load_dotenv

load_dotenv()


st.set_page_config(
    page_title="Document",
    page_icon="📃",
)

def load_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "wb") as f:
        f.write(file_content)
    layzer = UpstageLayoutAnalysisLoader(file_path, output_type="html")
    docs = layzer.load()
    return docs

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
    script = script_generator(docs)
    captions = caption_extractor(docs)
    mapping_data = caption_slide_mapper(script, captions)
    script








