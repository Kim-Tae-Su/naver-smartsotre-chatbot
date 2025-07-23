import streamlit as st
from rag import get_ai_response_stream
from dotenv import load_dotenv
load_dotenv()

# 페이지 제목, 아이콘 설정
st.set_page_config(
    page_title="네이버 스마트스토어 FAQ 챗봇",
    page_icon="🛍️"
)

# 페이지 상단 타이틀, 설명 표시
st.title("🛍️ 네이버 스마트스토어 FAQ 챗봇")
st.caption("네이버 스마트스토어의 자주 묻는 질문을 기반으로 정확한 답변을 제공합니다!")

# 세션 상태에 대화 내역이 없으면 초기화
if 'message_list' not in st.session_state:
    st.session_state.message_list = []

# 기존 대화 내역 화면에 출력
# 역할(user/assistant)에 따라 채팅 버블로 메시지 표시
for message in st.session_state.message_list:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# 채팅 입력창을 통해 사용자 입력 받기
# 사용자 메시지 화면에 표시 및 세션에 저장
# AI 응답을 스트리밍 방식으로 받아오기
# AI 메시지 화면에 표시 및 세션에 저장
if user_question := st.chat_input(placeholder="스마트스토어에 대해 궁금한 내용을 입력해 주세요!"):
    with st.chat_message("user"):
        st.write(user_question)
    st.session_state.message_list.append({"role": "user", "content": user_question})

    with st.spinner("답변을 생성하는 중입니다"):
        ai_response = get_ai_response_stream(user_question, st.session_state.message_list)
        with st.chat_message("ai"):
            ai_message = st.write_stream(ai_response)
        st.session_state.message_list.append({"role": "assistant", "content": ai_message})