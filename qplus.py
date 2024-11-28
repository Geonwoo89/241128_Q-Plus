import tempfile
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import streamlit as st

# Streamlit secrets에서 API 키 읽어오기
openai_api_key = st.secrets["openai"]["api_key"]

# PDF 파일을 업로드하고 인덱싱하는 함수
def load_and_index_pdfs(uploaded_files):
    all_texts = []
    metadatas = []  # 메타데이터 저장용 리스트
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name
            reader = PdfReader(temp_file_path)

            # PDF의 각 페이지를 처리
            for page_number, page in enumerate(reader.pages, start=1):
                text = page.extract_text()
                all_texts.append(text)

                # 메타데이터 추가: 문서명과 페이지 번호
                metadatas.append({"document_name": uploaded_file.name, "page_number": page_number})

    # 텍스트를 작은 덩어리로 나누기
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.create_documents(all_texts, metadatas=metadatas)

    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY를 설정하세요.")  # API 키 확인

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)  # API 키 전달
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store

# QA 체인 생성 함수
def create_qa_chain(vector_store):
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY를 설정하세요.")  # API 키 확인
    
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

            # 관련 문서와 페이지 번호 출력
            st.write("관련된 문서 및 페이지 번호:")
            results = vector_store.similarity_search_with_score(query, k=3)  # 상위 3개의 관련 문서 검색
            for result in results:
                metadata = result[0].metadata  # 메타데이터 추출
                st.write(f"문서명: {metadata['document_name']}, 페이지 번호: {metadata['page_number']}")

if __name__ == "__main__":
    main()