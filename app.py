"""
This code loads environment variables using the `dotenv` library and sets the necessary environment variables for Azure services.
The environment variables are loaded from the `.env` file in the same directory as this notebook.
"""
import os
from openai import AzureOpenAI
from dotenv import load_dotenv
from dotenv import dotenv_values

from langchain import hub
from langchain_openai import AzureChatOpenAI
from langchain_community.document_loaders import AzureAIDocumentIntelligenceLoader
from langchain_openai import AzureOpenAIEmbeddings
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.vectorstores.azuresearch import AzureSearch

if os.path.exists(".env"):
    load_dotenv(override=True)
    config = dotenv_values(".env")

# Set environment variables for Azure services
doc_intelligence_endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
doc_intelligence_key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")

azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_openai_chat_completions_deployment_name = os.getenv("AZURE_OPENAI_CHAT_COMPLETIONS_DEPLOYMENT_NAME")

azure_embedding_endpoint = os.getenv("AZURE_EMBEDDING_ENDPOINT")
azure_openai_embedding_model = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL")
embedding_vector_dimensions = os.getenv("EMBEDDING_VECTOR_DIMENSIONS")



# Load a document and split it into semantic chunks
# Initiate Azure AI Document Intelligence to load the document. You can either specify file_path or url_path to load the document.
loader = AzureAIDocumentIntelligenceLoader(file_path="<path to your file>", api_key = doc_intelligence_key, api_endpoint = doc_intelligence_endpoint, api_model="prebuilt-layout")
docs = loader.load()

# Split the document into chunks base on markdown headers.
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]
text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

docs_string = docs[0].page_content
splits = text_splitter.split_text(docs_string)

print("Length of splits: " + str(len(splits)))

# Embed and index the chunks
# Embed the splitted documents and insert into Azure Search vector store

aoai_embeddings = AzureOpenAIEmbeddings(
    azure_deployment="<Azure OpenAI embeddings model>",
    openai_api_version="<Azure OpenAI API version>",  # e.g., "2023-12-01-preview"
)

vector_store_address: str = os.getenv("AZURE_SEARCH_ENDPOINT")
vector_store_password: str = os.getenv("AZURE_SEARCH_ADMIN_KEY")

index_name: str = "<your index name>"
vector_store: AzureSearch = AzureSearch(
    azure_search_endpoint=vector_store_address,
    azure_search_key=vector_store_password,
    index_name=index_name,
    embedding_function=aoai_embeddings.embed_query,
)

vector_store.add_documents(documents=splits)

# Get the vector embedding for an input text
def get_embedding(text: str):
    
    openai_embedding_client = AzureOpenAI(
        azure_endpoint=azure_embedding_endpoint,
        api_key=azure_openai_api_key,
        api_version="2024-05-01-preview"
    )

    response = openai_embedding_client.embeddings.create(
        input=text,
        model=azure_openai_embedding_model,
    )

    embedding = response.data[0].embedding
    print(text)
    return embedding

# Retrive relevant chunks based on a question
def get_relevant_documents(question: str):
    """
    Retrieve relevant documents based on the question.
    """
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    return retriever.get_relevant_documents(question)
