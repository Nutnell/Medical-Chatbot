from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv

from src.prompt import prompt_template
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import Pinecone
from pinecone import Pinecone as PineconeClient
import boto3

from meta_llm import MetaBedrockLLM

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

load_dotenv()
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
INDEX_NAME = os.environ.get("INDEX_NAME", "medical-chatbot1")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-2")

# Validate API key
if not PINECONE_API_KEY:
    print("❌ Missing PINECONE_API_KEY in environment variables.")
    exit()

# Embeddings
try:
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": False},
    )
    print("✅ HuggingFaceEmbeddings initialized.")
except Exception as e:
    print(f"❌ Error initializing HuggingFaceEmbeddings: {e}")
    embeddings = None

# Pinecone setup
docsearch = None
try:
    pc = PineconeClient(api_key=PINECONE_API_KEY)
    print("✅ Pinecone client initialized.")
    if INDEX_NAME not in pc.list_indexes().names():
        raise RuntimeError(f"Pinecone index '{INDEX_NAME}' not found.")
    docsearch = Pinecone.from_existing_index(index_name=INDEX_NAME, embedding=embeddings)
    print(f"✅ Pinecone index '{INDEX_NAME}' loaded.")
except Exception as e:
    print(f"❌ Pinecone error: {e}")
    docsearch = None

# Bedrock + Meta LLM setup
llm = None
try:
    bedrock_runtime = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    llm = MetaBedrockLLM(
        bedrock_client=bedrock_runtime,
        model_id="arn:aws:bedrock:us-east-2:004642588306:inference-profile/us.meta.llama3-1-8b-instruct-v1:0",
        temperature=0.8,
        max_gen_len=256  # Slightly reduced to curb looping
    )
    print("✅ MetaBedrockLLM initialized.")
except Exception as e:
    print(f"❌ Failed to initialize Meta LLM: {e}")
    llm = None

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs = {"prompt": PROMPT}

qa_chain = None
if llm and docsearch:
    try:
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=docsearch.as_retriever(search_kwargs={"k": 2}),
            return_source_documents=True,
            chain_type_kwargs=chain_type_kwargs,
        )
        print("✅ QA chain initialized.")
    except Exception as e:
        print(f"❌ QA chain error: {e}")

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    msg = data.get("message")
    session_id = data.get("session_id")

    print(f"🗨️ Session {session_id} received: '{msg}'")

    if not msg:
        return jsonify({"answer": "Message cannot be empty."}), 400

    if qa_chain is None:
        print("⚠️ QA chain not ready.")
        return jsonify({"answer": "Medical Assistant is offline. Check backend logs."}), 503

    try:
        result = qa_chain.invoke({"query": msg})
        response_text = result.get("result", "Sorry, I couldn’t fetch a valid answer.")
        print(f"💬 Response: {response_text}")
        return jsonify({"answer": response_text})
    except Exception as e:
        print(f"❌ QA invocation error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"answer": f"Internal error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
