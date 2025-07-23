import os
from openai import OpenAI
from dotenv import load_dotenv

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from db import build_vector_db_if_needed
# 환경변수 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
assert OPENAI_API_KEY, "OPENAI_API_KEY is missing."

# OpenAI 클라이언트 생성
client = OpenAI(api_key=OPENAI_API_KEY)

# Chroma DB 연결
embedding_fn = OpenAIEmbeddingFunction(api_key=OPENAI_API_KEY, model_name="text-embedding-3-small")
collection = build_vector_db_if_needed()

# 독립 질문이 스마트스토어 FAQ 주제와 연관성이 있는지 판단하기 위해 사용자 질문만으로 임베딩을 생성한다.
# 벡터 DB에서 가장 유사한 FAQ 문서를 검색한다.
# 검색 결과의 거리가 임계값보다 작으면 True, 그렇지 않으면 False를 반환한다.
def is_standalone_question_valid_topic(question: str, threshold: float = 1.3):
    question_embedding = embedding_fn([question])
    results_q = collection.query(query_embeddings=question_embedding, n_results=1)
    distance_q = results_q["distances"][0][0]
    return distance_q < threshold

# 후속 질문이 FAQ 주제와 관련 있는지 판단하기 위해 직전 대화 이력과 사용자 질문을 하나의 쿼리로 합쳐 임베딩을 생성한다.
# 벡터 DB에서 가장 유사한 FAQ 문서를 검색한다.
# 검색 결과의 거리가 임계값보다 작으면 True, 그렇지 않으면 False를 반환한다.
def is_followup_question_valid_topic(question: str, messages: list, threshold: float = 1.3):
    history = "\n".join(
        f"{msg['role']}: {msg['content']}" for msg in messages if msg["role"] in ["user", "assistant"]
    )
    full_query = f"{history}\n\n사용자 질문: {question}"
    full_embedding = embedding_fn([full_query])
    results_full = collection.query(query_embeddings=full_embedding, n_results=1)
    distance_full = results_full["distances"][0][0]
    return distance_full < threshold

# 사용자 질문과 대화 이력을 받아 직전 대화 내용을 참고하여 질문이 독립적인 질문인지,
# 아니면 직전 대화에 이어지는 후속 질문인지 판단하기 위해 LLM에 분류 프롬프트를 보낸다.
# LLM의 분류 결과가 "STANDALONE"이면 True, "FOLLOWUP"이면 False를 반환한다.
def is_standalone(question: str, messages: list) -> bool:
    classification_prompt = f"""
        아래 문장이 독립적인 질문인지, 직전 대화에 이어지는 후속 질문인지 판단해 주세요.

        대화 이력:
        {messages[-2]['content'] if len(messages) >= 2 else ''}

        새 질문: "{question}"

        STANDALONE 또는 FOLLOWUP 만 출력하세요.
        """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "너는 문장 분류기야."},
            {"role": "user", "content": classification_prompt}
        ]
    )
    result = response.choices[0].message.content.strip()
    return result == "STANDALONE"

# 유효하지 않은 주제의 사용자 질문을 받아 스마트스토어 FAQ와 관련이 없음을 알리는 프롬프트를 작성한다.
# 해당 프롬프트를 포함한 메시지로 LLM에 요청한다.
# LLM이 스트리밍으로 생성한 답변 텍스트를 yield로 외부에 순차적으로 전달한다.
def get_invalid_topic_response(question: str):
    prompt = f"""
        사용자 질문:
        \"{question}\"

        이 질문은 스마트스토어 FAQ와 직접 관련이 없다.
        따라서 반드시 아래 형식으로만 답변한다:

        첫 줄: "저는 스마트 스토어 FAQ를 위한 챗봇입니다. 스마트 스토어에 대한 질문을 부탁드립니다."
        그 아래: 위 FAQ 예시에서 {question}과 관련해 유저가 궁금할 만한 질문을 2가지만 제안해 주세요.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "너는 네이버 스마트스토어 FAQ 챗봇입니다."},
            {"role": "user", "content": prompt}
        ],
        stream=True
    )

    for chunk in response:
        content = chunk.choices[0].delta.content  
        if content:
            yield content

# 유효한 주제의 사용자 질문과 대화 이력을 받아 질문 임베딩을 생성하고 벡터 DB에서 관련 FAQ 문서를 검색한디.
# 검색된 문서들을 하나의 컨텍스트로 합쳐 프롬프트를 작성하고 해당 프롬프트를 포함한 메시지로 LLM에 요청한다.
# LLM이 스트리밍으로 생성한 답변 텍스트를 yield로 외부에 순차적으로 전달한다.
def get_valid_topic_response(question: str, messages: list):
    query_embedding = embedding_fn([question])
    results = collection.query(query_embeddings=query_embedding, n_results=3)
    context_chunks = results["documents"]
    flat_contexts = []
    for docs in context_chunks:
        if isinstance(docs, list):
            flat_contexts.extend(docs)
        else:
            flat_contexts.append(docs)
    context_text = "\n\n".join(flat_contexts)
    
    prompt = f"""
        다음은 네이버 스마트스토어 FAQ의 일부입니다:

        \"\"\"
        {context_text}
        \"\"\"

        위 FAQ 내용을 참고하여 사용자 질문에 답변해 주세요:
        질문: \"{question}\"

        답변을 짧고 친절하게 작성 해주세요. 추가로 관련 질문 2개를 제안해 주세요.
    """

    updated_messages = (
        [{"role": "system", "content": "너는 네이버 스마트스토어 FAQ 챗봇입니다."}]
        + messages
        + [{"role": "user", "content": prompt}]
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=updated_messages,
        stream=True
    )
    
    for chunk in response:
        content = chunk.choices[0].delta.content  
        if content:
            yield content

# 사용자의 질문과 대화 이력을 받아 질문이 독립적인지 판별하고 질문 주제가 유효한지 검사한다.
# 주제에 따라 적절한 AI 응답 스트림을 생성한다.
def get_ai_response_stream(question: str, messages: list):
    is_standalone_question = is_standalone(question, messages)
    if is_standalone_question:
        is_valid = is_standalone_question_valid_topic(question)
    else:
        is_valid = is_followup_question_valid_topic(question, messages)
    if not is_valid:
        yield from get_invalid_topic_response(question)
    else:
        yield from get_valid_topic_response(question, messages)
