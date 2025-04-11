from flask import Flask, render_template, jsonify, request
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from src.prompt import system_prompt, system_prompt_citation
from src.helper import download_hugging_face_embeddings
import os
from flask_cors import CORS

app = Flask(__name__)

CORS(app)

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

embeddings = download_hugging_face_embeddings()
index_name = "medical-chatbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

llm = ChatGoogleGenerativeAI(google_api_key=GOOGLE_API_KEY, model="gemini-2.0-flash")

# Initialize conversation memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Modified prompt to include chat history and request citations
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("system", "Previous conversation:\n{chat_history}"),
    ("human", "{input}")
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route("/")
def index():
    return render_template("chat.html")

# @app.route("/get", methods=["GET", "POST"])
# def chat():
#     msg = request.form["msg"]
#     print(msg)
#     response = rag_chain.invoke({"input": msg})
#     print("Response: ", response["answer"])
#     return str(response["answer"])

# @app.route("/api/chat", methods=["POST"])
# def chat():
#     msg = request.json.get("msg")  # note: JSON input from React
#     print(msg)
#     response = rag_chain.invoke({"input": msg})
#     print("Response: ", response["answer"])
#     return jsonify({"answer": response["answer"]})
@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    msg = data.get("msg")
    print("User message:", msg)

    # Get the response from the RAG chain
    response = rag_chain.invoke({
        "input": msg,
        "chat_history": memory.load_memory_variables({})["chat_history"]
    })
    
    # Save the interaction to memory
    memory.save_context({"input": msg}, {"output": response["answer"]})
    
    # Get source documents
    source_docs = response.get("context", [])
    
    # Format citations
    new_line = "\n"
    citation_string = f"Top related sources: {new_line}"

    for i, doc in enumerate(source_docs, 1):
        metadata = doc.metadata
        source = metadata.get("source", "Unknown source")
        page = metadata.get("page", "Unknown page")
    
        split_source = source.split("/")[1]
        citation_string = f"{citation_string} [Source: {split_source}, Page: {round(page)}] {new_line}"
    
    print("Bot response:", response["answer"])
    print("Citations:", citation_string)

    response["answer"] = f"{response['answer']} {new_line} {new_line} {citation_string}"
    return jsonify({
        "answer": response["answer"],
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
