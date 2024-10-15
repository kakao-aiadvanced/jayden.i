import os
import bs4
from langchain import hub
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# API 키 설정
os.environ["OPENAI_API_KEY"] = ""
# os.environ["LANGSMITH_API_KEY"] = "your_langsmith_api_key"

url = "https://applied-llms.org/"
loader = WebBaseLoader(
    web_paths=(url,),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("content")
        )
    ),
)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)
vectorstore = Chroma.from_documents(documents=chunks, embedding=OpenAIEmbeddings())

# 사용자 쿼리
user_query = "Could you explain LLMs"

# 관련성 평가 및 시스템 프롬프트 작성
relevance_checker_prompt = """
You are a system that evaluates the relevance of retrieved text chunks to the user query.
Please evaluate whether the retrieved chunk is relevant to the user query.

Output should be a JSON with a single key "relevance" and the value should be either "yes" if the chunk is relevant or "no" if the chunk is not relevant.

User query: "{query}"
Retrieved chunk: "{chunk}"
"""

parser = JsonOutputParser()

def check_relevance(query, chunk):
    prompt = relevance_checker_prompt.format(query=query, chunk=chunk)
    llm = ChatOpenAI(model_name="gpt-4o-mini")

    # invoke 메서드 호출
    response = llm.invoke(prompt)

    # AIMessage 객체에서 content 속성 추출
    response_content = response.content  # 여기서 response.content 사용

    # JSON 파서 사용하여 응답 파싱
    try:
        parsed_response = parser.parse(response_content)  # 응답 문자열로 파싱
        return parsed_response

    except Exception as e:
        print(f"Error parsing relevance response: {e}")
        return {"relevance": "no"}  # 기본값으로 'no' 반환

# 검색 쿼리에 따라 가장 관련성이 높은 청크 검색
retrieved_chunks = vectorstore.similarity_search(user_query, k=5)

# 중복 제거
unique_chunks = {chunk.page_content: chunk for chunk in retrieved_chunks}.values()

# 루프를 사용하여 검색된 청크의 관련성 평가
relevance_results = []
for chunk in retrieved_chunks:
    result = check_relevance(user_query, chunk.page_content)
    relevance_results.append(result)

# 결과 평가 및 답변 생성
for result in relevance_results:
    if result["relevance"] == "no":
        print("NO")
        # 디버깅 로직 추가 (예: Chunk size, overlap 등)
        break
    else:
        # 관련성이 있는 경우 답변 생성
        answer_prompt = f"Based on the retrieved chunk, generate an answer for the user query: {user_query}. Retrieved chunk: {chunk.page_content}"
        llm = ChatOpenAI(model_name="gpt-4o-mini")
        answer = llm(answer_prompt)

        # 환각(hallucination) 평가
        hallucination_checker_prompt = f"""
        You are a system that evaluates whether the generated answer contains hallucination.
        Please evaluate whether the generated answer is accurate or contains hallucination.

        Output should be a JSON with a single key "hallucination" and the value should be either "yes" if hallucination is present or "no" if it is not present.

        Generated answer: "{answer}"
        """

        hallucination_result = check_relevance("hallucination check", answer)

        # 환각 평가 결과에 따라 처리
        if hallucination_result.get("hallucination") == "yes":
            # 최대 2번까지 다시 생성
            for _ in range(2):  # 최대 2회 시도
                answer = llm(answer_prompt)  # 답변을 다시 생성
                hallucination_result = check_relevance("hallucination check", answer)
                print("Re-evaluated Hallucination result:", hallucination_result)
                # hallucination_result가 "no"로 바뀌면 반복문 종료
                if hallucination_result.get("hallucination") == "no":
                    break
        else:
            # hallucination_result가 "no"일 경우
            print("No hallucination detected, proceeding with the answer.")

        # 답변과 출처 출력
        print("Generated Answer:", answer)
        print("Source:", chunk.metadata.get("source", "Unknown"))