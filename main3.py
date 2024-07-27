import os
import anthropic
from halo import Halo
import pprint
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_anthropic import ChatAnthropic
from langchain.memory import ConversationBufferMemory
from langchain.embeddings.base import Embeddings
from langchain.chains import ConversationalRetrievalChain
import numpy as np

load_dotenv()
pp=pprint.PrettyPrinter(indent=4)

client = anthropic.Anthropic(api_key=os.getenv("CLAUDE_KEY"))
langchain_anthropic = ChatAnthropic(
    model=os.getenv("MODEL_NAME"),
    anthropic_api_key=os.getenv("CLAUDE_KEY")
)

class AnthropicEmbeddings(Embeddings):
    def __init__(self, client):
        self.client = client
    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]
    def embed_query(self, text):
        response = self.client.completions.create(
            model="claude-2.0",
            prompt=f"{anthropic.HUMAN_PROMPT} please generate an embedding for the following text: {text}{anthropic.AI_PROMPT}",
            max_tokens_to_sample=1000,
        )

        return np.random.rand(1536)

def process_pdf(pdf_path):
    print("Loading PDF...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"PDF loaded.Number of pages: {len(documents)}")

    print("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    print(f"Text Split into{len(texts)} chunks ")

    print("Creating Embeddings and Vector Store...")
    embeddings = AnthropicEmbeddings(client)

    vectorstore = FAISS.from_documents(texts,embeddings)
    print("Vector Store Created")
    
    return vectorstore

def setup_conversational_chain(vectorstore):
    memory = ConversationBufferMemory(
        llm=langchain_anthropic,
        max_token_limit=1000,
        memory_key="chat_history",
        return_messages=True
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=langchain_anthropic,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

    return chain

def generate_response(chain, user_message):
    spinner = Halo(text='Loading...', spinner='dots')
    spinner.start()

    response = chain({"Question: user_message"})

    spinner.stop()
    print("Request: ")
    pp.pprint(user_message)
    print("Response: ")
    pp.pprint(response['answer'])

    return response['answer']

def main():
    pdf_path ="ck.pdf"
    print(f"Starting to process PDF: {pdf_path}")
    print("Checking if file exists...")
    if not os.path.exists(pdf_path):
        print(f"Error: file '{pdf_path}' not found")
        return
    try:
        vectorstore = process_pdf(pdf_path)
        print("PDF Processed Successfully.")
        chain = setup_conversational_chain(vectorstore)
        print("Conversinf chain setup.")
        print(f"PDF  '{pdf_path}' processed. you can now start ")
        while True:
            input_text = input("You: ")
            if input_text.lower() =="quit":
                break
            response = generate_response(chain, input_text)
            print(f"Claude: {response}")
    except Exception as e:
        print(f"An Error Occured: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__== "__main__":
    main()