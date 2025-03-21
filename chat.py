import streamlit as st
from dotenv import load_dotenv

from llm import get_ai_message

load_dotenv()

st.set_page_config(page_title="소득세 Chatbot", page_icon="뭔데이건")

st.title("소득세 챗봇")  # h1
st.caption("소득세관한 모든 것을 답함")



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

