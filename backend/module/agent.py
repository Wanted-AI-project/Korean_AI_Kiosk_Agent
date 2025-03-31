# backend/chatbot/agent.py

from dotenv import load_dotenv
import os
import re
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain_community.utilities import SQLDatabase
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import Tool
from langchain.tools.retriever import create_retriever_tool
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
# from database import get_db_connection

load_dotenv()

embedding = OpenAIEmbeddings()
vector_store = Chroma(persist_directory="../chroma_db", embedding_function=embedding)
llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
sql_llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)
db = SQLDatabase.from_uri("sqlite:///../data/Comfile_Coffee_DB.db")
# db = get_db_connection()

store = {}  # 세션별 대화 저장소

prompt = ChatPromptTemplate.from_messages([
    ("system",
            "You are a helper agent who is built into the kiosk and tries to take orders."
            "Please be courteous and helpful to customers' questions and orders. "
            "You should check out their beverage options like hot and iced."
            "You should check out their other options like payment method."
            "However, please only answer questions related to cafe work, such as inquiries about orders and menus."

            "The ordering process is as follows: Select menu - Confirm takeout - Confirm additional menus or inquiries -Final confirmation of order with price information -Select payment method - Wait for payment - After payment, guidance message and 'Thank_you' is sent."

            "Always use the provided tools to answer."
            "Use chat_retriever_tool for questions related to order history. "
            "Answer all questions about orders and menus using menu_db_sql_tool."
            "In particular, please use menu_db_sql_tool to answer about price information."
            "you must answer in conversational form. Without any decoration such as symbols or dictionary forms."
            "Do not guess; retrieve data using the tools before responding."
            "Don't say you don't have it, just use the tools we provide."
            "Nnevertheless ,you don't know the answer, just say that you don't know."
            "Always reference the provided data when generating a response. Do not fabricate or infer any information."
            "Answer in Korean."),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

def make_query_and_run(text: str):
    info = db.get_table_info()
    custom_prompt = PromptTemplate(
        input_variables=["query"],
        partial_variables={'info': info},
        template="""
        You are an expert in SQL and databases. Generate an SQL query based on the given question. 

        - The question may not always provide the exact menu name, so extract only the key characteristics.
        - If multiple keywords are provided, process them using `AND`.
        - Use `LIKE` and `%` to handle partial matches.
        - If the question includes "price" or "가격", do not return only the price—return all related information by using `SELECT *`.
        - Convert "따뜻한" or "뜨거운" to "HOT" and "차가운" or "아이스" to "ICE" in SQL queries.

        Return only the raw SQL query without any markdown formatting, code blocks, or explanations.

        You must reference the following database structure:
        {info}



        question: {query}
        """
    )
    sql_chain = custom_prompt | sql_llm
    result_query = sql_chain.invoke(text).content
    result = re.sub(r"```sql\n(.*?)\n```", r"\1", result_query, flags=re.DOTALL).strip()
    return db.run(result)

vector_retriever = None  # 초기화 시 설정

def initialize_session(session_id: str):
    global vector_retriever, store
    store[session_id] = ChatMessageHistory()
    vector_retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, 'lambda_mult': 0.1, "filter": {"session_id": session_id}}
    )

def get_session_history(session_id: str):
    if session_id not in store: # session_id 가 store에 없는 경우
        # 새로운 ChatMessageHistory 객체를 생성하여 store에 저장
        store[session_id] = ChatMessageHistory()
    return store[session_id]    # 해당 세션 ID에 대한 세션 기록 반환

def save_session_to_vector_db(session_id: str):
    if session_id in store:
        chat_history = store[session_id].messages
        for msg in chat_history:
            vector_store.add_texts([msg.content], metadatas=[{"session_id": session_id, "role": msg.type}])

def clear_session(session_id: str):
    if session_id in store:
        del store[session_id]

def get_agent_with_tools():
    menu_db_sql_tool = Tool(
        name="menu_db_sql_tool",
        func=make_query_and_run,
        description="You can search order information from Database"
    )
    chat_retriever_tool = create_retriever_tool(
        vector_retriever,
        name='chat_search',
        description='use this tool to search chat history and order history'
    )
    tools = [menu_db_sql_tool, chat_retriever_tool]
    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
    agent_exe = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=8,
        must_use_tools=[menu_db_sql_tool]
    )
    return RunnableWithMessageHistory(
        agent_exe,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
