# In store_index.py
from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain_community.vectorstores import Pinecone
import pinecone
from dotenv import load_dotenv
import os
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from langchain_pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_REGION = os.environ.get("PINECONE_REGION")
PINECONE_HOST = os.environ.get("PINECONE_HOST")
PINECONE_CLOUD = os.environ.get("PINECONE_CLOUD")

extracted_data = load_pdf("data/")

text_chunks = text_split(extracted_data)
print(f"Number of text chunks: {len(text_chunks)}")

embeddings = download_hugging_face_embeddings()

model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": False}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)

embedding_dimension = 384

# Initialize the Pinecone client
pc = PineconeClient(api_key=PINECONE_API_KEY)
# Define index name
index_name = "medical-chatbot1"

# Check if index exists and create if not
if index_name not in pc.list_indexes().names():
    print(f"Creating Pinecone index: {index_name} with dimension {embedding_dimension}")
    pc.create_index(
        name=index_name,
        dimension=embedding_dimension,
        metric="cosine",
        # Use ServerlessSpec directly
        spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
    )
    print("Index created. Waiting for index to be ready...")
    import time

    time.sleep(60)

# LangChain's Pinecone class to add texts
pinecone_vectorstore = Pinecone(
    index_name=index_name,
    embedding=embeddings,
    pinecone_api_key=PINECONE_API_KEY,
)

# Creating Embeddings for each of the Text Chunks and storing it
docsearch = pinecone_vectorstore.add_texts([t.page_content for t in text_chunks])

print("Embeddings created and added to Pinecone index.")
