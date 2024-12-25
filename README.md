# GenAI_AWSBedrock_opensearch
Gen AI utility AWS bedrock model,RAG (Opensearch)

Main Steps
# Tech Statck 
1. Data Ingestion: Load data to an Opensearch Index
2. Embedding and Model: AWS Bedrock Titan
3. Vector Store and Endpoint: Opensearch
4. LLM model : anthropic.claude-3-sonnet-20240229-v1:0
5. Cloud : AWS Bedrock & AWS opensearch for Vector storage
6. Evaluating ,Monitoring and metrics : Langsmith
7. Data: PDF data
8. LangChain & AWS Langchain  is a framework for building applications powered by large language models (LLMs). It is designed to streamline the integration of LLMs with various components, such as external data sources, user-defined workflows, and tools like databases or APIs. LangChain helps developers build robust, scalable, and dynamic AI-driven applications.
9. Microservice: FastAPI & Uvicorn

# Project Set Up

1. The Python version used for this project is Python 3.11.

Clone the repo (or download it as a zip file):

git clone [https://github.com/benitomartin/aws-bedrock-opensearch-langchain.git](https://github.com/anjijava16/GenAI_AWSBedrock_opensearch.git)

```
Create the virtual environment named main-env using Conda with Python version 3.10:

conda create -n main-env python=3.11
conda activate main-env
```

2. Install the requirements.txt:

pip install -r requirements.txt

3. Run the FastAPI code
   python rag_api_chat.py

4. Open below Microservice URL

   http://0.0.0.0:8847/docs#/default/chat_application_chat_post

5. Sample Requset :
    ```
   {
  "query": "Can you describe the React approach?",
  "user_id": "string"
  }
  ```

6. Resposne
```

```
