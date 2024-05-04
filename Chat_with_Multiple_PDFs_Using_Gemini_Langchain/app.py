import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai

from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Setup embedding path and model name
faiss_index_path = "vector_embeddings/faiss_index"
embedding_model = "models/embedding-001"

# Environment variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

# Configure the google API key from environment variables
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        # Read all pages from the pdf
        pdf_reader= PdfReader(pdf)
        # Pages from the PDF would be in the form of list
        for page in pdf_reader.pages:
            # Extract text and append to text
            text+= page.extract_text()
    return  text

# Break the text into chunks
def get_text_chunks(text):
    # Create a splitter object
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)

    # Break text into chunks
    chunks = text_splitter.split_text(text)

    return chunks



# Convert the text chunks into vector embeddings
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    print(type(embeddings))
    
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local(faiss_index_path)



# Create conversational chain
def get_conversational_chain():
    prompt_template = """
    You are a pirate Q&A assisstant whose job is to answer questions in as much detail as possible using the provided context. If
    you cannot find any context, just say "Answer is not available in the context". Don't hallucinate and generate wrong asnwers\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    # Create a model object
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.4, apiKey=google_api_key)

    # Create the prompt template
    prompt = PromptTemplate(template=prompt_template, input_variables=["context","question"])

    # Create chain
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain
    

def user_input(user_question):
    # Create the embeddings object
    embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model, apiKey=google_api_key)

    # When we upload the PDF's, they are converted to embeddings and stored locally.
    # We will not load those embeddings in a new variables
    new_db = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)

    # ch: The vector embedding of the user's query is then compared against the vector embeddings 
    # of documents or other data in a database or index. A similarity metric, such as cosine 
    # similarity or Euclidean distance, is used to identify the data points (e.g. documents, images, 
    # etc.) that are most similar to the query vector.
    docs = new_db.similarity_search(user_question)

    # Create the conversational chain
    chain = get_conversational_chain()


    response = chain(
        {"input_documents":docs, "question":user_question},
        return_only_outputs = True
    )

    print(response)
    st.write("Reply: \n\n", response["output_text"] )


# Create the streamlit app
def main():
    st.set_page_config("Chat with multiple PDFs")
    st.header("Chat with PDF's using Gemini 💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    # If user question has been entered, then the conversational chain should get created  
    if user_question:
        user_input(user_question)

    # Create the sidebar for uploading PDF's
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                # Get input file text
                raw_text = get_pdf_text(pdf_docs)

                # Break down text into chunks
                text_chunks = get_text_chunks(raw_text)

                # Convert into embedding vectors and store locally
                get_vector_store(text_chunks)
                st.success("Done")






if __name__ == "__main__":
    main()