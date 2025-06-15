import os
from dotenv import load_dotenv, dotenv_values

from openai import AzureOpenAI
from langchain import hub
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.document_loaders import AzureAIDocumentIntelligenceLoader
from langchain.schema import StrOutputParser
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import AzureSearch
from langchain.chains import RetrievalQA

# === 1. Load environment variables ===
if os.path.exists(".env"):
    load_dotenv(override=True)
    config = dotenv_values(".env")
else:
    raise FileNotFoundError("Missing .env file in the current directory.")

# === 2. Validate and load required configs ===
def get_env_var(name):
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value

# Azure service credentials
doc_intel_endpoint = get_env_var("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
doc_intel_key = get_env_var("AZURE_DOCUMENT_INTELLIGENCE_KEY")

aoai_endpoint = get_env_var("AZURE_OPENAI_ENDPOINT")
aoai_key = get_env_var("AZURE_OPENAI_API_KEY")
aoai_deployment_name = get_env_var("AZURE_OPENAI_CHAT_COMPLETIONS_DEPLOYMENT_NAME")

embedding_deployment_name = get_env_var("AZURE_OPENAI_EMBEDDING_MODEL")
embedding_endpoint = get_env_var("AZURE_EMBEDDING_ENDPOINT")

vector_store_endpoint = get_env_var("AZURE_SEARCH_ENDPOINT")
vector_store_key = get_env_var("AZURE_SEARCH_ADMIN_KEY")
embedding_dims = int(get_env_var("EMBEDDING_VECTOR_DIMENSIONS"))

index_name = "catering-docs-index"
file_path = "catering docs.pdf"

# === 3. Load and parse document ===
print("Loading document from:", file_path)

loader = AzureAIDocumentIntelligenceLoader(
    file_path=file_path,
    api_key=doc_intel_key,
    api_endpoint=doc_intel_endpoint,
    api_model="prebuilt-layout"
)
docs = loader.load()

if not docs or not docs[0].page_content.strip():
    raise ValueError("Document content is empty or failed to load.")

print("Loaded document successfully.")

# === 4. Split document using markdown headers ===
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]
#text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

splits = text_splitter.split_documents(docs)

for i, doc in enumerate(splits):
    print(f"\nChunk {i+1}:")
    print(doc.page_content)

print(f"Document split into {len(splits)} chunks.")

# === 5. Initialize embedding model ===
aoai_embeddings = AzureOpenAIEmbeddings(
    azure_deployment=embedding_deployment_name,
    azure_endpoint=embedding_endpoint,
    openai_api_key=aoai_key,  # Correct arg name now
    openai_api_version="2024-05-01-preview"
)


# === 6. Initialize Azure Cognitive Search vector store ===
vector_store = AzureSearch(
    azure_search_endpoint=vector_store_endpoint,
    azure_search_key=vector_store_key,
    index_name=index_name,
    embedding_function=aoai_embeddings.embed_query,
)

# Optional: add documents only if not already indexed
print("Indexing chunks to Azure Cognitive Search...")
vector_store.add_documents(documents=splits)
print("Indexing complete.")

# === 7. Create retriever and LLM for Q&A ===
retriever = vector_store.as_retriever(
    search_type="similarity",
    k=3  # ✅ Use `k` directly instead of search_kwargs
)


llm = AzureChatOpenAI(
    azure_deployment=aoai_deployment_name,
    azure_endpoint=aoai_endpoint,
    api_key=aoai_key,
    api_version="2024-05-01-preview",
    temperature=0
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# === 8. Ask a question ===
def ask_question(question: str):
    result = qa_chain.invoke({"query": question})
    print("\nQUESTION:", question)
    print("ANSWER:", result["result"])
    print("\nSOURCE DOCUMENTS:")
    for doc in result["source_documents"]:
        print("-", doc.metadata.get("source", "Unknown"))

# === 9. Example Usage ===
ask_question("What catering services are offered?")
