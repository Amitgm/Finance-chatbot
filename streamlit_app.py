import streamlit as st
from langchain_core.runnables import RunnableLambda
from mysql.connector import errorcode
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough 
from langchain.chains import create_sql_query_chain
from langchain.sql_database import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_ollama import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.schema.runnable import RunnableParallel
import os
from dotenv import load_dotenv
from langchain.document_loaders import JSONLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import HumanMessage

# Initialize memory
memory = ConversationBufferMemory(return_messages=True)

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")

model_name = "sentence-transformers/all-mpnet-base-v2"

embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cuda"}
    )



# THE SQL CHAIN
def run_full_sql_chain():
    def get_schema(_):
        return db.get_table_info()

    llm = ChatOllama(temperature=0.4, model="llama3.1")
    uri = "mysql+mysqlconnector://root:root@localhost:3306/stocks"
    db = SQLDatabase.from_uri(uri)

    template = """
    You are a Financial AI assistant. Based on the table schema below, write a SQL query that would answer the user's question.
    <SCHEMA>{schema}</SCHEMA>
    Conversation History: {chat_history}
    Write only the SQL query and nothing else.
    Question: {question}
    SQL Query:"""
    prompt = ChatPromptTemplate.from_template(template)

    sql_chain = (
        RunnablePassthrough.assign(schema=get_schema) 
        | prompt 
        | llm.bind(stop=["\nSQL Result:"]) 
        | StrOutputParser()
    )

    template = """
    You are a data analyst at a company. Based on the table schema below, question, sql query, and sql response, write a natural language response.
    <SCHEMA>{schema}</SCHEMA>
    Conversation History: {chat_history}
    SQL Query: <SQL>{query}</SQL>
    User question: {question}
    SQL Response: {response}"""
    prompt = ChatPromptTemplate.from_template(template)

    full_chain = (
        RunnablePassthrough.assign(query=sql_chain).assign(
            schema=lambda _: db.get_table_info(),
            response=lambda vars: db.run(vars["query"]),
            chat_history=lambda _: memory.load_memory_variables({})["history"]
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    return full_chain

# THE SENTIMENT RAG CHAIN
def rag_full_chain():


    llm = ChatOllama(temperature=0.4, model="llama3.1")
    
    qa_prompt = PromptTemplate(
        template="""
        You are a financial AI assistant analyzing stock news sentiment.
        <context>{context}</context>
        Question: {input}""",
        input_variables=["context", "input"]
    )

    loader = JSONLoader(
        file_path="sentiments.json",
        jq_schema='.[] | {symbol: .symbol, metric: .news[]}',
        text_content=False
    )

    docs = loader.load()

    
    # embeddings = OllamaEmbeddings(model="nomic-embed-text")

    vectorstore = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_sentiment_store")

    retriever = vectorstore.as_retriever()

    document_chain = create_stuff_documents_chain(llm, qa_prompt)

    return create_retrieval_chain(retriever, document_chain)

# THE FINANCE DATA CHAIN
def financial_data_chain():
    llm = ChatOllama(model="llama3.1")
    
    qa_prompt = PromptTemplate(
        template="""
        Extract EXACT numerical values from this financial data:
        <context>{context}</context>
        Question: {input}
        Format: "ANSWER: $X" (just the number with dollar sign)""",
        input_variables=["context", "input"]
    )

    loader = JSONLoader(
        file_path="financial_explained.json",
        jq_schema='.[] | {symbol: .symbol, fiscalYear: .lastFiscalYearEnd, metric: .metrics[]}',
        text_content=False
    )
    docs = loader.load()

    # embeddings = OllamaEmbeddings(model="nomic-embed-text")

    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever()
    document_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(retriever, document_chain)

def my_custom_sql_analysis(result):

    prompt = f"""
        ACT AS A FINANCIAL ANALYST. Analyze this stock data if the data is given with thorough numbers else ignore:

        {result}

        PROVIDE:
        1. Most volatile day (largest price swing)
        2. Key support/resistance levels (from lows/highs)
        3. Volume analysis (correlation with price moves)
        4. Bullish/bearish signals

        FORMAT RESPONSE AS:
        - Summary
        - Key Metrics
        - Technical Analysis
    """

    llm = ChatOllama(temperature = 0.4, model="llama3.1")

    llm = ChatOllama(model="llama3.1", temperature=0)
    messages = [HumanMessage(content=prompt)]

    response = llm(messages)

    return response



# Initialize chains
sql_chain = run_full_sql_chain()

# my_custom_sql_analysis()

rag_sentiment_chain = rag_full_chain()
rag_finance_info_chain = financial_data_chain()

# Prepare inputs
chat_history = memory.load_memory_variables({})["history"]

prepare_inputs = RunnableLambda(lambda inputs: {
    "sql_inputs": {
        "question": inputs["input"],
        "chat_history": chat_history,
    },
    "rag_sentiment_inputs": {
        "input": inputs["input"]
    },
    "rag_finance_data_inputs": {
        "input": inputs["input"]
    }
})



combined_chain = (
    prepare_inputs
    | RunnableLambda(lambda inputs: {
        "sql_result": sql_chain.invoke(inputs["sql_inputs"]),
        "rag_sentiment_result": rag_sentiment_chain.invoke(inputs["rag_sentiment_inputs"]),
        "rag_finance_data_result": rag_finance_info_chain.invoke(inputs["rag_finance_data_inputs"]),
    })
    | RunnableLambda(lambda inputs: {
        "custom_sql_analysis": my_custom_sql_analysis(inputs["sql_result"]),
        **inputs  # keep all previous keys too
    })
    | RunnableLambda(lambda inputs: {
        "final_answer": f"""
        **Database Analysis:**

        {inputs['custom_sql_analysis']}
        
        **Market Sentiment:**
        {inputs['rag_sentiment_result']['answer']}
        
        **Financial Metrics:**
        {inputs['rag_finance_data_result']['answer']}
        """
    })
)


# Streamlit App
def main():
    st.title("Financial Assistant Chatbot")
    st.markdown("Ask questions about stocks and financial data")
    
    if "messages" not in st.session_state:

        st.session_state.messages = []
    
    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask your question..."):

        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):

            st.markdown(prompt)
        
        
        
        with st.spinner("Analyzing..."):
            try:

                response = combined_chain.invoke({"input": prompt})

                final_answer = response["final_answer"]
                
                with st.chat_message("assistant"):

                    st.markdown(final_answer)
                
                st.session_state.messages.append({"role": "assistant", "content": final_answer})

                memory.save_context({"input": prompt}, {"output": final_answer})
                
            except Exception as e:

                error_msg = f"Error: {str(e)}"
                
                with st.chat_message("assistant"):
                    st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    main()