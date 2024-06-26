
import streamlit as st
from streamlit_chat import message
from langdetect import detect
from transformers import MarianMTModel, MarianTokenizer

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, AutoModelForSeq2SeqLM
from transformers import pipeline,TextStreamer
from langchain import HuggingFacePipeline, PromptTemplate
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings

MODEL_NAME = 'Intel/neural-chat-7b-v3-1'

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, bnb_4bit_compute_dtype=torch.float16, load_in_4bit=True
)
generation_config = GenerationConfig.from_pretrained(MODEL_NAME)
generation_config.max_new_tokens = 1024
generation_config.temperature = 0.0001
generation_config.do_sample = True



streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

text_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=1024,
    temperature=0,
    top_p=0.95,
    repetition_penalty=1.15,
    streamer=streamer,
)
     

#load the pdf files from the path
loader = DirectoryLoader('data/',glob="*.pdf",loader_cls=PyPDFLoader)  # Langchain.documents_Loaders .
documents = loader.load()


#split text into chunks
text_splitter  = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50) # from Langchain.Text_splitter
text_chunks = text_splitter.split_documents(documents)

#create embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", #model for creating embeddings
                                   model_kwargs={'device':"cpu"}) #model_kwrgs specifications for using CPU, use CUDA for GPU

#vectorstore
vector_store = FAISS.from_documents(text_chunks,embeddings)  # Faiss vectore store

DEFAULT_SYSTEM_PROMPT ="""You are Whale ai, a helpful and respectful assistant dedicated to providing accurate information about seafood, marine life, and the services of MPEDA. Always assist with care, respect, and truth. Respond with utmost utility yet securely. Avoid harmful, unethical, prejudiced, or negative content. Ensure replies promote fairness and positivity.

If a question is related to seafood trade, MPEDA services, marine life, or aquatic topics, provide detailed and accurate information. If a question does not make any sense, or is not factually coherent within these topics, explain why instead of answering something not correct. If you don't know the answer to a seafood-related question, please don't share false information.

For questions outside the realm of seafood and MPEDA, such as those about space exploration or politics or any other, respond in a polite and light-hearted manner, encouraging the user to steer the conversation back to seafood and MPEDA. For example, you might say, "While I'm quite the aficionado of the vast ocean, I'm afraid my knowledge doesn't extend to the vastness of space. Let's dive back into the world of seafood and marine life, shall we?"
""".strip()


def generate_prompt(prompt: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
    return f"""
[INST] <>
{system_prompt}
<>

{prompt} [/INST]
""".strip()

SYSTEM_PROMPT = "Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer."

template = generate_prompt(
    """
{context}

Question: {question}
""",
    system_prompt=SYSTEM_PROMPT,
)

prompt = PromptTemplate(template=template, input_variables=["context", "question"])

llm = HuggingFacePipeline(pipeline=text_pipeline, model_kwargs={"temperature": 0})

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt},
)

# # Function to translate text
# def translate(text, model_name):
#     tokenizer = MarianTokenizer.from_pretrained(model_name)
#     model = MarianMTModel.from_pretrained(model_name)
#     translated = model.generate(**tokenizer(text, return_tensors="pt", padding=True))
#     return tokenizer.decode(translated[0], skip_special_tokens=True)


tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-mul-en")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-mul-en")

from transformers import pipeline
translator = pipeline("translation", tokenizer=tokenizer, model=model, max_length=1024)

def translate_to_english(text,lang):
    translation = translator(text,src_lang=lang)
    return translation

model2 = 'Helsinki-NLP/opus-mt-en-mul'
tokenizer2 = MarianTokenizer.from_pretrained(model2)
translation_model2 = MarianMTModel.from_pretrained(model2)

from transformers import pipeline
translator = pipeline("translation", tokenizer=tokenizer, model=model, max_length=1024)

def translate_from_english(text,lang):
    translation = translator(text,tgt_lang=lang,src_lang='en')
    return translation

# ######

st.title("WHALE AI, ask questions about exporter's prodcuts, Seafood and much more. Talk in your native language, let us surprise you.")


def conversation_chat(user_input):
    if user_input:
        with st.spinner('Processing...'):
            try:
                # Detect language
                lang = detect(user_input)
            
                translated_input = user_input
                # If not English, translate to English
                if lang != 'en':
                    input = translate_to_english(user_input,lang)
                    translated_input = input[0]['translation_text']

                # Get response from chatbot
                response = qa_chain(translated_input)
                answer = response["result"]

                # If input was translated, translate response back to original language
                if lang != 'en':
                    response = translate_from_english(answer, lang)
                    answer= response[0]['translation_text']
                    
                # st.write(answer)
                st.session_state['history'].append((user_input, answer))
            except Exception as e:
                st.error(f"An error occurred: {e}")
    return answer

def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! have any questions regarding MPEDA exporters, rules, quality control or seafood in general? Ask away.. "]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey! 👋"]

def display_chat_history():
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Your Questions Here", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            output = conversation_chat(user_input)

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")


# Initialize session state
initialize_session_state()
# # Display chat history
# display_chat_history()


import streamlit as st
import pandas as pd

def save_user_info():
    """Page to collect and save user information."""
    st.title("User Information")
    with st.form("user_info_form"):
        name = st.text_input("Name")
        email = st.text_input("Email")
        submit = st.form_submit_button("Submit")

        if submit and name and email:  # Ensure both name and email are provided
            new_entry = pd.DataFrame([{"Name": name, "Email": email}])
            try:
                df = pd.read_csv('user_info.csv')
                df = pd.concat([df, new_entry], ignore_index=True)
            except FileNotFoundError:
                df = new_entry

            df.to_csv('user_info.csv', index=False)
            st.success("Information saved!")


# # Initialize session state
# initialize_session_state()

# Page Selector in Sidebar
page = st.sidebar.selectbox("Choose a page", ["Chatbot", "User Information"])

if page == "Chatbot":
    display_chat_history()
elif page == "User Information":
    save_user_info()


initialize_session_state()