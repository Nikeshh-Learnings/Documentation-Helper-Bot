import os
from dotenv import load_dotenv
from typing import Any, List, Dict
from langchain_openai import OpenAIEmbeddings
from langchain_openai.chat_models.base import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores.pinecone import Pinecone as langchainPinecone
from pinecone import Pinecone
from consts import INDEX_NAME

load_dotenv()

pc = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY"),
    environment=os.environ["PINECONE_ENVIRONMENT_REGION"],
)

def run_llm(query: str, chat_history: List[Dict[str, Any]] = []) -> Any:
    embeddings = OpenAIEmbeddings()
    docsearch = langchainPinecone.from_existing_index(
        index_name=INDEX_NAME, embedding=embeddings
    )
    chat = ChatOpenAI(verbose=True, temperature=0)

    qa = ConversationalRetrievalChain.from_llm(
        llm=chat, retriever=docsearch.as_retriever(), return_source_documents=True
    )

    return qa({"question": query, "chat_history": chat_history})


if __name__ == "__main__":
    print(run_llm(query="What is LangChain?"))