import os
import tempfile
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS  # 변경된 임포트 경로
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import streamlit as st

# PDF 파일을 업로드하고 인덱싱하는 함수
def load_and_index_pdfs(uploaded_files):
    all_texts = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name
            reader = PdfReader(temp_file_path)
            for page in reader.pages:
                all_texts.append(page.extract_text())

    # 텍스트를 작은 덩어리로 나누기
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.create_documents(all_texts)

    # 환경 변수에서 API 키 읽기
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY 환경 변수를 설정하세요.")  # 환경 변수 체크

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)  # API 키 전달
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store

# QA 체인 생성 함수
def create_qa_chain(vector_store):
    # 환경 변수에서 API 키 읽기
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY 환경 변수를 설정하세요.")  # 환경 변수 체크
    
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        openai_api_key=openai_api_key
    )
    retriever = vector_store.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
    return qa_chain

# Streamlit 애플리케이션 메인 함수
def main():
    st.title("PDF 질문 답변 챗봇")

    # PDF 파일 업로드 기능
    uploaded_files = st.file_uploader("PDF 파일을 업로드하세요", accept_multiple_files=True, type="pdf")

    if uploaded_files:
        # 업로드된 파일들을 인덱싱
        vector_store = load_and_index_pdfs(uploaded_files)
        qa_chain = create_qa_chain(vector_store)

        # 질문 입력 받기
        query = st.text_input("질문을 입력하세요:")

        if query:
            # 질문에 대한 답변 실행
            response = qa_chain.run(query)
            st.write("답변:", response)

if __name__ == "__main__":
    main()
