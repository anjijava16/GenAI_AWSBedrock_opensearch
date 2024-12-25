from enum import Enum

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

app = FastAPI(
    title="Chat Memory API",
    description="""ChatApplication.""",
    version="0.0.1",
    debug=True,
)

import os
import sys

import boto3
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"

# logger configuration
logger.remove()
logger.add(sys.stdout, level=os.getenv("LOG_LEVEL", "INFO"))


def bedrock_embeddings(bedrock_client, bedrock_embedding_model_id):
    """
    Create a LangChain vector embedding using Bedrock.

    Args:
        bedrock_client (boto3.client): The Bedrock client.
        bedrock_embedding_model_id (str): The ID of the Bedrock embedding model.

    Returns:
        BedrockEmbeddings: A LangChain Bedrock embeddings client.

    """
    logger.info(f"Creating LangChain vector embedding using Bedrock model: {bedrock_embedding_model_id}")
    return BedrockEmbeddings(
        client=bedrock_client,
        model_id=bedrock_embedding_model_id)


def opensearch_vectorstore(index_name, bedrock_embeddings_client, _is_aoss=False):
    """
    Create an OpenSearch vector search client.

    Args:
        index_name (str): The name of the OpenSearch index.
        opensearch_password (str): The password for OpenSearch authentication.
        bedrock_embeddings_client (BedrockEmbeddings): The Bedrock embeddings client.
        opensearch_endpoint (str): The OpenSearch endpoint URL.
        _is_aoss (bool, optional): Whether it's Amazon OpenSearch Serverless. Defaults to False.

    Returns:
        OpenSearchVectorSearch: An OpenSearch vector search client.

    """
    CLUSTER_URL = 'https://localhost:9200'
    logger.info(f"Creating OpenSearch vector search client for index: {index_name}")
    username = 'admin'
    password = 'admin'
    # http_auth=(username, password),
    # verify_certs=False
    return OpenSearchVectorSearch(
        index_name=index_name,
        embedding_function=bedrock_embeddings_client,
        opensearch_url=CLUSTER_URL,
        http_auth=(username, password),
        verify_certs=False,
        is_aoss=_is_aoss,
        timeout=30,
        retry_on_timeout=True,
        max_retries=5,
    )


def bedrock_llm(bedrock_client, bedrock_model_id):
    """
    Create a Bedrock language model client.

    Args:
        bedrock_client (boto3.client): The Bedrock client.
        bedrock_model_id (str): The ID of the Bedrock model.

    Returns:
        ChatBedrock: A LangChain Bedrock chat model.

    """
    logger.info(f"Creating Bedrock LLM with model: {bedrock_model_id}")

    model_kwargs = {
        # "maxTokenCount": 4096,
        "temperature": 0,
        "topP": 0.3,
    }
    chat = ChatBedrock(
        model_id=bedrock_model_id,
        model_kwargs={"temperature": 0.1},
    )
    return chat


class InputRequest(BaseModel):
    query: str = None
    user_id: str = None


@app.post("/chat")
def chat_application(req: InputRequest):
    """
    Main function to run the LangChain with Bedrock and OpenSearch workflow.

    This function sets up the necessary clients, creates the LangChain components,
    and executes a query using the retrieval chain.
    """
    logger.info("Starting the LangChain with Bedrock and OpenSearch workflow...")

    # bedrock_model_id = "amazon.titan-text-lite-v1"
    bedrock_model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
    bedrock_embedding_model_id = "amazon.titan-embed-text-v1"
    region = "us-east-1"
    index_name = "rag"
    domain_name = "rag"
    # question = " Can you describe the React approach?"
    question = req.query

    logger.info(
        f"Creating Bedrock client with model {bedrock_model_id}, and embeddings with {bedrock_embedding_model_id}")

    # Creating all clients for chain
    bedrock_client = boto3.client("bedrock-runtime", region_name=region)
    llm = bedrock_llm(bedrock_client, bedrock_model_id)

    embeddings = bedrock_embeddings(bedrock_client, bedrock_embedding_model_id)

    vectorstore = opensearch_vectorstore(index_name, embeddings)

    print(f"vectorstore={vectorstore}")

    prompt = ChatPromptTemplate.from_template("""You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    Use five sentences maximum.
    
    {context}

    Question: {input}
    Answer:""")

    chain = create_stuff_documents_chain(llm, prompt)

    print(f"chain ={chain}")

    retrieval_chain = create_retrieval_chain(
        retriever=vectorstore.as_retriever(),
        combine_docs_chain=chain
    )

    response = retrieval_chain.invoke({"input": question})

    logger.info(f"The answer from Bedrock {bedrock_model_id} is: {response.get('answer')}")

    print(f" type of response ={response}")
    return {
        "response": response.get('answer'),
        "rag_response": response
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8847)
