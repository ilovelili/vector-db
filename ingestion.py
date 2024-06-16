import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

# https://www.pinecone.io/
# document loaders implementations: https://github.com/langchain-ai/langchain/tree/master/libs/langchain/langchain/document_loaders
# text splitters: https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/
# text embeddings: https://python.langchain.com/v0.1/docs/modules/data_connection/text_embedding/
if __name__ == "__main__":
    print("Ingesting...")
    loader = TextLoader("./mediumblog1.txt")
    document = loader.load()

    print("splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

    print("ingesting...")
    PineconeVectorStore.from_documents(
        texts, embeddings, index_name=os.environ["INDEX_NAME"]
    )
    print("finish")
