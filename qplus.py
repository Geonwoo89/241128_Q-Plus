import tempfile
import pdfplumber
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import streamlit as st
import io
from concurrent.futures import ThreadPoolExecutor

# Streamlit secrets에서 API 키 읽어오기
openai_api_key = st.secrets["openai"]["api_key"]

# 병렬 처리를 위한 함수
def process_pdf_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

        # PDFPlumber로 텍스트 추출
        texts = []
        metadatas = []

        with pdfplumber.open(temp_file_path) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()

                if not text:
                    text = "[빈 페이지]"

                texts.append(text)
                metadatas.append({
                    "document_name": uploaded_file.name,
                    "page_number": page_number
                })

        return texts, metadatas

# PDF 파일을 업로드하고 인덱싱하는 함수
def load_and_index_pdfs(uploaded_files):
    all_texts = []
    metadatas = []

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_pdf_file, uploaded_files))

    # 병렬 처리된 결과 합치기
    for texts, metadata in results:
        all_texts.extend(texts)
        metadatas.extend(metadata)

    # 텍스트를 작은 덩어리로 나누기
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.create_documents(all_texts, metadatas=metadatas)

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
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
        with st.spinner("파일 처리 중..."):
            vector_store = load_and_index_pdfs(uploaded_files)
        st.success("처리 완료!")

        # QA 체인 생성
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