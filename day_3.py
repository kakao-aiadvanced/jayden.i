import streamlit as st
import os
from langchain_openai import ChatOpenAI
from tavily import TavilyClient
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.graph import END, StateGraph
from typing import List
from typing_extensions import TypedDict
from langchain_core.documents import Document

tavily = TavilyClient(api_key="")

st.set_page_config(
    page_title="Research Assistant",
    page_icon=":orange_heart:",
)

os.environ['OPENAI_API_KEY'] = ""
llm = ChatOpenAI(model="gpt-4o-mini", temperature = 0)


class GraphState(TypedDict):
    def __init__(self):
        self.websearch_attempted = False  # websearch 시도를 기록할 변수
        self.generate_attempted = False  # generate 시도를 기록할 변수

    question: str
    generation: str
    web_search: str
    documents: List[Document]


def main():
    # 모델 선택
    llm_model = st.sidebar.selectbox(
        "Select Model",
        options=[
            "llama3",
        ],
    )

    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]

    # 웹 페이지 로드 및 문서 분할
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs_list)

    # 벡터 저장소 생성 (docs_retrieval)
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    )
    retriever = vectorstore.as_retriever()

    # 쿼리 입력
    def query(state):
        # 사용자로부터 질문을 입력받습니다.
        user_question = "what is llm?"

        # 입력된 질문을 state의 question 키에 저장합니다.
        state["question"] = user_question

        # 상태를 반환 (선택적)
        return state

    # RAG 에이전트 노드 및 엣지 정의
    def retrieve(state):
        print("---RETRIEVE---")
        question = state["question"]
        documents = retriever.invoke(question)
        return {"documents": documents, "question": question}

    def generate(state):
        print("---GENERATE---")

        if state.generate_attempted:
            print("failed: hallucination")
            return END  # 종료

        state.generate_attempted = True

        question = state["question"]
        documents = state["documents"]

        # RAG 체인을 통해 생성
        generation = rag_chain.invoke({"context": documents, "question": question})

        # 출처 URL 및 제목 추가 (예시: 문서에서 가져오는 로직)
        source_urls = [doc["url"] for doc in documents if "url" in doc]  # URL 추출
        titles = [doc["title"] for doc in documents if "title" in doc]  # 제목 추출

        # 사용자에게 줄 답변 포맷 (출처 및 제목 포함)
        answer_to_user = {
            "generation": generation,
            "source_urls": source_urls,
            "titles": titles,
        }

        return {
            "documents": documents,
            "question": question,
            "generation": generation,
            "answer_to_user": answer_to_user,  # 사용자에게 줄 답변 추가
        }

    def answerToUser(state):
        print("---ANSWER TO USER---")

        generation = state.get("generation", "")
        source_urls = state.get("answer_to_user", {}).get("source_urls", [])
        titles = state.get("answer_to_user", {}).get("titles", [])
        question = state["question"]

        # 사용자에게 답변 출력
        print(f"Question: {question}")
        print(f"Answer: {generation}")

        if titles and source_urls:
            print("Sources:")
            for title, url in zip(titles, source_urls):
                print(f" - Title: {title}, URL: {url}")

        return END  # 종료

    def grade_documents(state):
        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]
        filtered_docs = []
        web_search = "No"
        for d in documents:
            score = retrieval_grader.invoke(
                {"question": question, "document": d.page_content}
            )
            grade = score["score"]
            if grade.lower() == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                web_search = "Yes"
                continue
        return {
            "documents": filtered_docs,
            "question": question,
            "web_search": web_search,
        }

    def web_search(state):
        print("---WEB SEARCH---")

        if state.websearch_attempted:
            print("failed: not relevant")
            return END  # 종료

        # 검색 시도 기록
        state.websearch_attempted = True

        question = state["question"]
        documents = state["documents"]
        docs = tavily.search(query=question)["results"]
        web_results = "\n".join([d["content"] for d in docs])
        web_results = Document(page_content=web_results)
        if documents is not None:
            documents.append(web_results)
        else:
            documents = [web_results]
        return {"documents": documents, "question": question}

    def grade_generation_v_documents_and_question(state):
        print("---CHECK HALLUCINATIONS---")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]

        score = hallucination_grader.invoke(
            {"documents": documents, "generation": generation}
        )
        grade = score["score"]

        if grade == "yes":
            print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            print("---GRADE GENERATION vs QUESTION---")
            score = answer_grader.invoke(
                {"question": question, "generation": generation}
            )
            grade = score["score"]
            if grade == "yes":
                print("---DECISION: GENERATION ADDRESSES QUESTION---")
                return "useful"
            else:
                print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                return "not useful"
        else:
            print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
            return "not supported"

    # RAG 에이전트 그래프 구성
    workflow = StateGraph(GraphState)
    workflow.add_node("query", query)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("hallucination Checker", grade_generation_v_documents_and_question)
    workflow.add_node("answerToUser", answerToUser)

    workflow.add_node("websearch", web_search)

    # query 노드에 대한 조건부 진입점 설정
    workflow.set_conditional_entry_point(
        query,  # query 노드 이름
        {
            "start": "query"  # 'start'라는 조건으로 시작할 때 'query' 노드로 진입
        }
    )

    # query 노드에서 retrieve 노드로 이동하는 전이 추가
    workflow.add_transition("query", "retrieve")
    # retrieve 노드에서 grade_documents 노드로 이동하는 전이 추가
    workflow.add_transition("retrieve", "grade_documents")

    # 조건부 흐름 설정
    # grade_documents 노드에서 'yes'일 경우 generate로 이동
    workflow.add_transition("grade_documents", "generate", condition="yes")

    # grade_documents 노드에서 'no'일 경우 websearch로 이동
    workflow.add_transition("grade_documents", "websearch", condition="no")

    # generate 노드에서 grade_generation_v_documents_and_question 노드로 이동하는 전이 추가
    workflow.add_transition("generate", "grade_generation_v_documents_and_question")

    # grade_generation_v_documents_and_question 노드에서 'yes'일 경우 generate로 이동
    workflow.add_transition("grade_generation_v_documents_and_question", "answerToUser", condition="yes")

    # grade_generation_v_documents_and_question 노드에서 'no'일 경우 generate로 이동
    workflow.add_transition("grade_generation_v_documents_and_question", "generate로", condition="no")


    app = workflow.compile()

    # rag_chain 정의
    llm = ChatOllama(model=llm_model, temperature=0)
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for question-answering tasks.
        Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.
        Use three sentences maximum and keep the answer concise <|eot_id|><|start_header_id|>user<|end_header_id|>
        Question: {question}
        Context: {context}
        Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["question", "context"],
    )
    rag_chain = prompt | llm | StrOutputParser()

    # retrieval_grader, hallucination_grader, answer_grader 정의
    llm = ChatOllama(model=llm_model, format="json", temperature=0)

    retrieval_grader_prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing relevance
        of a retrieved document to a user question. If the document contains keywords related to the user question,
        grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
        Provide the binary score as a JSON with a single key 'score' and no premable or explanation.
         <|eot_id|><|start_header_id|>user<|end_header_id|>
        Here is the retrieved document: \n\n {document} \n\n
        Here is the user question: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """,
        input_variables=["question", "document"],
    )
    retrieval_grader = retrieval_grader_prompt | llm | JsonOutputParser()

    hallucination_grader_prompt = PromptTemplate(
        template=""" <|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether
        an answer is grounded in / supported by a set of facts. Give a binary 'yes' or 'no' score to indicate
        whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a
        single key 'score' and no preamble or explanation. <|eot_id|><|start_header_id|>user<|end_header_id|>
        Here are the facts:
        \n ------- \n
        {documents}
        \n ------- \n
        Here is the answer: {generation}  <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["generation", "documents"],
    )
    hallucination_grader = hallucination_grader_prompt | llm | JsonOutputParser()

    answer_grader_prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether an
        answer is useful to resolve a question. Give a binary score 'yes' or 'no' to indicate whether the answer is
        useful to resolve a question. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
         <|eot_id|><|start_header_id|>user<|end_header_id|> Here is the answer:
        \n ------- \n
        {generation}
        \n ------- \n
        Here is the question: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["generation", "question"],
    )
    answer_grader = answer_grader_prompt | llm | JsonOutputParser()

    question_router_prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an expert at routing a
    user question to a vectorstore or web search. Use the vectorstore for questions on LLM  agents,
    prompt engineering, and adversarial attacks. You do not need to be stringent with the keywords
    in the question related to these topics. Otherwise, use web-search. Give a binary choice 'web_search'
    or 'vectorstore' based on the question. Return the a JSON with a single key 'datasource' and
    no premable or explanation. Question to route: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["question"],
    )
    question_router = question_router_prompt | llm | JsonOutputParser()

    answer_to_user_prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant providing a user with a detailed answer to their question. Include the generated answer, source URLs, and titles.
    Provide the response in a structured format.

    <|eot_id|><|start_header_id|>user<|end_header_id|> Here is the answer:
    \n ------- \n
    {generation}
    \n ------- \n
    Here are the sources:
    \n Titles: {titles}
    \n URLs: {source_urls}
    \n Here is the question: {question}
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["generation", "source_urls", "titles", "question"],
    )

    answer_to_user = answer_to_user_prompt | llm | JsonOutputParser()

    # ----------------------------------------------------------------------
    # Streamlit 앱 UI
    st.title("Research Assistant powered by OpenAI")

    input_topic = st.text_input(
        ":female-scientist: Enter a topic",
        value="Superfast Llama 3 inference on Groq Cloud",
    )

    generate_report = st.button("Generate Report")

    if generate_report:
        with st.spinner("Generating Report"):
            inputs = {"question": input_topic}
            for output in app.stream(inputs):
                for key, value in output.items():
                    print(f"Finished running: {key}:")
            final_report = value["generation"]
            st.markdown(final_report)

    st.sidebar.markdown("---")
    if st.sidebar.button("Restart"):
        st.session_state.clear()
        st.experimental_rerun()


main()