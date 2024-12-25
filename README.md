# GenAI_AWSBedrock_opensearch
Gen AI utility AWS bedrock model,RAG (Opensearch)


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

## The Python version used for this project is Python 3.11.


## . Clone the repo (or download it as a zip file):

  git clone [https://github.com/benitomartin/aws-bedrock-opensearch-langchain.git](https://github.com/anjijava16/GenAI_AWSBedrock_opensearch.git)


```
Create the virtual environment named main-env using Conda with Python version 3.10:

conda create -n main-env python=3.11
conda activate main-env

```

## Install the requirements.txt:

   pip install -r requirements.txt

##  Run the FastAPI code
   
   python rag_api_chat.py

##  Open below Microservice URL
   http://0.0.0.0:8847/docs#/default/chat_application_chat_post


##  Sample Requset 
   
```
   {
  "query": "Can you describe the React approach?",
  "user_id": "string"
  }
  ```

##  Resposne

```
{
  "response": "Based on the context provided, I can describe the ReAct (Reasoning and Acting) approach in the following way:\n\n1) ReAct is a method that allows large language models to synergize reasoning and acting for tasks that require multi-step reasoning and decision making in interactive environments.\n\n2) It prompts the language model with sparse thoughts or reasoning steps along with the observations and actions taken in an interactive environment. This allows the model to integrate its reasoning process with the actions and observations in a coherent stream of inputs.\n\n3) For example, in a question-answering task, ReAct would prompt the model with thoughts like \"I need to search for X\", followed by the search results, then \"The observation says Y, so the answer is Z\". This interleaves the reasoning process with the actions taken (searching) and observations received.\n\n4) ReAct was evaluated on multi-hop question-answering tasks like HotpotQA, fact-checking tasks, and interactive decision-making environments like ALFWorld (a text-based game) and WebShop (an online shopping website). It showed superior performance compared to baselines like just prompting actions or using chain-of-thought prompting.\n\n5) The key advantage of ReAct is that it allows the language model to perform interpretable multi-step reasoning integrated with actions taken in an interactive environment, leading to better performance on complex reasoning and decision-making tasks."

 
}

```
