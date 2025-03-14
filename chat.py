import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain import hub
import os

load_dotenv()
st.set_page_config(page_title="소득세 Chatbot", page_icon="뭔데이건")

st.title("소득세 챗봇")  # h1
st.caption("소득세관한 모든 것을 답함")

embedding = OpenAIEmbeddings(model='text-embedding-3-large')
index_name = "tax-index"

database = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embedding)
dictionary = ["사람을 나타내는 표현 -> 거주자"]


llm = ChatOpenAI(model='gpt-4o')
prompt = hub.pull("rlm/rag-prompt")
retriever = database.as_retriever(search_kwars={'k': 4})

qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type_kwargs={"prompt": prompt}, chain_type="stuff")


def get_ai_message(user_message):
    prompt = ChatPromptTemplate.from_template(f"""
        사용자의 질문을 보고, 우리의 사전을 참고해 사용자의 질문을 변경해주세요.
        만약 변경할 필요가 없다고 판단된다면, 사용자의 질문을 변경하지 않아도 됩니다.
        그런 경우 질문만 리턴해주세요.
        사전: {dictionary}

        질문: {{question}}
    """)

    dictionary_chain = prompt | llm | StrOutputParser()
    tax_chain = {"query": dictionary_chain} | qa_chain
    ai_message = tax_chain.invoke({"question": user_message})

    return ai_message["result"]


if 'message_list' not in st.session_state:
    st.session_state.messge_list = []

for message in st.session_state.messge_list:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if user_question := st.chat_input(placeholder="질문 내용 입력"):
    with st.chat_message("user"):
        st.write(user_question)

    st.session_state.messge_list.append({"role": "user", "content": user_question})

    with st.spinner("답변을 생성하는 중"):
        ai_message = get_ai_message(user_question)
        with st.chat_message("ai"):
            st.write(ai_message)

    st.session_state.messge_list.append({"role": "user", "content": ai_message})

