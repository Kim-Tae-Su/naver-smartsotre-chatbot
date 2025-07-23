import os
import pickle

import pysqlite3
import sys
sys.modules["sqlite3"] = pysqlite3
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv
import tiktoken
# 환경 변수 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
assert OPENAI_API_KEY, "OPENAI_API_KEY is missing."

# 네이버 스토어 FAQ 데이터 로드 함수
def load_faq_data():
    with open('./final_result.pkl', 'rb') as f:
        data = pickle.load(f)
    return data

# 텍스트를 스플릿해서 청크 리스트를 리턴하는 함수
def split_text_with_overlap(text, chunk_size, chunk_overlap):
    encoder = tiktoken.encoding_for_model("gpt-4o-mini")
    tokens = encoder.encode(text)
    total = len(tokens)

    chunks = []
    pos = 0

    while pos < total:
        end = min(pos + chunk_size, total)
        chunk = encoder.decode(tokens[pos:end])
        chunks.append(chunk)
        pos += chunk_size - chunk_overlap

    return chunks
# 벡터 DB를 구축하는 함수
# ChromaDB를 사용하여 FAQ 데이터를 벡터화하고 저장
# 컬렉션을 생성하거나 이미 존재하면 가져옴
# 임베딩 함수도 같이 연결하여 자동 임베딩 기능 활성화
# 컬렉션에 문서, ID, 임베딩 벡터 저장
def build_vector_db_if_needed():
    chroma_client = chromadb.Client()
    embedding_fn = OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-3-small"
    )

    collection = chroma_client.get_or_create_collection(
        name="naver-smartstore-faq",
        embedding_function=embedding_fn
    )

    if len(collection.get(include=['documents'])['documents']) == 0:
        data = load_faq_data()
        full_text = '\n'.join(data)
        chunks = split_text_with_overlap(full_text, 1500, 50)
        ids = [str(i) for i in range(len(chunks))]
        embeddings = embedding_fn(chunks)
        collection.add(documents=chunks, ids=ids, embeddings=embeddings)

    return collection