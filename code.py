# ================================================================
# Enhanced Microcontroller RAG + LINE Bot (Improved Retrieval)
# ================================================================

import os
from dotenv import load_dotenv

# Load environment variables from .env when running locally
load_dotenv()
import re
import sys
import subprocess
from typing import List, Dict, Any

# ================================================================
#  HARDCODED TOKENS (DEV ONLY)
# ================================================================
LINE_CHANNEL_ACCESS_TOKEN = os.getenv('LINE_CHANNEL_ACCESS_TOKEN', "yrYvQZcWle/bU98wyTd8sTw/8huLe1KBovhUL01d0w7MeTltSme+d5XD9V1GRe3mAVnzVVH2GYPIFABhCoOcpITMoAt0iWI8EorycBOLPEFFYEezeYZJPkXOLv5VHRy2ilgplELce3zwirEOKRYZrgdB04t89/1O/w1cDnyilFU=")
LINE_CHANNEL_SECRET = os.getenv('LINE_CHANNEL_SECRET', "15e7919c700c93820c5f94cecaee32d8")
NGROK_AUTHTOKEN = os.getenv('NGROK_AUTHTOKEN', "321Sf8mdH5pVmdJpwAGO25yrGY0_3Bh9AmgNpxyeGapkYtHT8")

# ================================================================
# Imports
# ================================================================
from flask import Flask, request, abort
from pyngrok import ngrok

from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import (
    Configuration, ApiClient, MessagingApi,
    ReplyMessageRequest, TextMessage
)
from linebot.v3.webhooks import MessageEvent, TextMessageContent

from langchain_community.document_loaders import TextLoader
from langchain.schema import Document
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma

from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever

from langchain_cohere import CohereRerank

from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Used to securely store your API key
# Removed google.colab.userdata usage; secrets are loaded from environment (.env) via python-dotenv

# ================================================================
# Ensure rank_bm25 available
# ================================================================
def _ensure_rank_bm25():
    try:
        import rank_bm25  # noqa: F401
    except Exception:
        try:
            print("Attempting to install rank_bm25...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "rank_bm25"])
            print("rank_bm25 installed successfully.")
        except Exception as e:
            print(f"[WARN] install rank_bm25 failed: {e}")

# ================================================================
# Enhanced Query Preprocessing
# ================================================================
def preprocess_question(q: str) -> str:
    """Enhanced query preprocessing for better retrieval"""
    low = q.lower()

    # Remove common question words that don't add search value
    stop_words = ['คือ', 'อะไร', 'ไหม', 'หรือไม่', 'บ้าง', 'มี', 'สามารถ']
    words = q.split()
    filtered_words = [w for w in words if w.lower() not in stop_words]

    # Add domain-specific expansions
    expansions = []

    # Mapping for better retrieval - Enhanced for project question
    keyword_mapping = {
        r"(สอบปฏิบัติ|โครงงาน|การสอบ|project|assessment|ข้อกำหนด|เกณฑ์)": "โครงงานกลุ่มรายวิชา Microcontroller แบ่งกลุ่ม 5-6 คน เก็บคะแนน 20 คะแนน เกณฑ์การให้คะแนน ข้อกำหนด รายละเอียดโครงงาน พรีเซนต์ 30 นาที รายงาน 20 หน้า PDF",
        r"(ไม่มีพื้นฐาน|ขาดพื้นฐาน|พื้นฐานอิเล็กทรอนิกส์|ไม่มีความรู้)": "ไม่มีพื้นฐานอิเล็กทรอนิกส์ สามารถเรียน Microcontroller ได้หรือไม่ ผลกระทบของการขาดความรู้",
        r"(อุปกรณ์|ใช้อะไร|components|ตัวอย่าง)": "อุปกรณ์ที่ใช้ในวงจร Microcontroller input output devices เครื่องซักผ้า เครื่องปรับอากาศ",
        r"(ต่อยอด|เรียนต่อ|advanced|วิชาไหน)": "ต่อยอดไปสู่วิชาใด Embedded Systems IoT Robotics Communication Networking",
        r"(เขียนโค้ด|programming|ภาษา|ฝึก)": "ฝึกการเขียนโค้ดภาษาโปรแกรมเบื้องต้น C programming ตัวแปร ชนิดข้อมูล",
        r"(วิชา.*microcontroller|microcontroller.*คือ)": "Microcontroller คืออะไร อุปกรณ์อิเล็กทรอนิกส์ คอมพิวเตอร์ขนาดเล็ก ควบคุมการทำงาน",
    }

    original_query = ' '.join(filtered_words) if filtered_words else q

    for pattern, expansion in keyword_mapping.items():
        if re.search(pattern, low):
            expansions.append(expansion)

    if expansions:
        return f"{original_query} {' '.join(expansions)}"

    return original_query

# ================================================================
# Initialize Enhanced RAG chain
# ================================================================
def initialize_rag_chain(md_file: str = "micro_rag_optimized.md"):
    print("⏳ Loading and indexing documents...")
    if not os.path.exists(md_file):
        raise FileNotFoundError(f"ไม่พบไฟล์ {md_file}")

    loader = TextLoader(md_file, encoding="utf-8")
    docs = loader.load()
    full_text = docs[0].page_content

    # Enhanced document splitting strategy
    # First split by headers to maintain context
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
        ],
        strip_headers=False
    )

    header_docs = header_splitter.split_text(full_text)

    # Then use recursive splitter for size control
    rc_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # Smaller chunks for better precision
        chunk_overlap=200,  # More overlap for context preservation
        separators=["\n\n", "\n", "。", ".", " ", ""]
    )

    final_docs = rc_splitter.split_documents(header_docs)

    # Add more metadata to help with retrieval
    for i, doc in enumerate(final_docs):
        doc.metadata['chunk_id'] = i
        # Extract keywords from content for better matching
        content_lower = doc.page_content.lower()
        keywords = []
        if 'โครงงาน' in content_lower or 'project' in content_lower or 'สอบปฏิบัติ' in content_lower:
            keywords.append('โครงงาน')
        if 'ไม่มีพื้นฐาน' in content_lower or 'ขาดความรู้' in content_lower:
            keywords.append('ไม่มีพื้นฐาน')
        if 'ต่อยอด' in content_lower or 'วิชา' in content_lower:
            keywords.append('ต่อยอด')
        if 'อุปกรณ์' in content_lower or 'devices' in content_lower:
            keywords.append('อุปกรณ์')
        # Convert the list of keywords to a comma-separated string
        doc.metadata['keywords'] = ', '.join(keywords) if keywords else None

    print(f"✅ Document split into {len(final_docs)} enhanced chunks.")

    # Load API keys
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    if GOOGLE_API_KEY is None:
        print("[WARN] GOOGLE_API_KEY is missing. Please set it in your environment variables.")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )

    vectorstore = Chroma.from_documents(
        documents=final_docs,
        embedding=embeddings
    )
    print("✅ Enhanced ChromaDB vector store created.")

    # Enhanced retriever configuration
    vect_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 20,  # Get more candidates
            "fetch_k": 80,  # Increase fetch pool
            "lambda_mult": 0.7  # Balance diversity vs relevance
        }
    )

    # BM25 with enhanced settings
    _ensure_rank_bm25()
    bm25_retriever = BM25Retriever.from_documents(final_docs)
    bm25_retriever.k = 20

    # Ensemble with adjusted weights - favor BM25 for exact keyword matching
    hybrid_retriever = EnsembleRetriever(
        retrievers=[vect_retriever, bm25_retriever],
        weights=[0.6, 0.4]  # Slightly favor BM25 for Thai text
    )

    # Enhanced reranking
    try:
        COHERE_API_KEY = os.getenv('COHERE_API_KEY')
        if COHERE_API_KEY is None:
            print("[WARN] COHERE_API_KEY is missing. Please set it in your environment variables.")
        compressor = CohereRerank(
            model="rerank-multilingual-v3.0",
            top_n=10,  # Keep more results for better coverage
            cohere_api_key=COHERE_API_KEY
        )
        retrieval = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=hybrid_retriever
        )
        print("✅ Using Enhanced Cohere Reranker.")
    except Exception as e:
        print(f"Could not initialize Cohere Reranker: {e}")
        retrieval = hybrid_retriever

    # Enhanced LLM and Prompt
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    if GOOGLE_API_KEY is None:
        print("[WARN] GOOGLE_API_KEY is missing. Please set it in your environment variables.")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        convert_system_message_to_human=True,
        google_api_key=GOOGLE_API_KEY
    )

    # Improved prompt template
    template = """คุณคือผู้เชี่ยวชาญไมโครคอนโทรลเลอร์ ตอบคำถามโดยใช้เฉพาะข้อมูลจากเอกสาร

กฎการตอบ:
- ใช้เฉพาะข้อมูลจาก Context
- ตอบตรงประเด็น กระชับ เข้าใจง่าย
- ห้ามใส่เครื่องหมาย * หรือ bullet points แบบ *
- ห้ามเขียน "จากเอกสารที่ให้มา" หรือ "(ข้อมูลที่ X)"
- ใช้รูปแบบย่อหน้าธรรมดา หรือขึ้นบรรทัดใหม่ด้วย - เท่านั้น
- หากไม่พบข้อมูลให้ตอบ "ไม่พบข้อมูลในเอกสาร"

Context:
{context}

คำถาม: {question}

คำตอบ:"""

    ANSWER_PROMPT = PromptTemplate.from_template(template)

    def format_docs(docs: List[Document]) -> str:
        formatted = []
        for i, doc in enumerate(docs, 1):
            content = doc.page_content.strip()
            if content:
                # Clean up all formatting issues
                content = re.sub(r'\*+\s*\*+', '', content)  # Remove multiple asterisks
                content = re.sub(r'^\*+\s*', '', content, flags=re.MULTILINE)  # Remove leading asterisks
                content = re.sub(r'\*+\s*$', '', content, flags=re.MULTILINE)  # Remove trailing asterisks
                content = re.sub(r'\*+([^*]+)\*+', r'\1', content)  # Remove asterisks around text
                content = re.sub(r'\n\s*\*\s*', '\n', content)  # Remove bullet asterisks
                content = re.sub(r'จากเอกสารที่ให้มา[,:\s]*', '', content)  # Remove reference phrases
                content = re.sub(r'\(ข้อมูลที่\s*\d+[,\s]*\d*\)', '', content)  # Remove data references
                formatted.append(content)
        return "\n\n".join(formatted[:10])  # Increase to 10 chunks for better coverage

    rag_chain = (
        {
            "context": retrieval,
            "question": RunnablePassthrough(),
        }
        | RunnablePassthrough.assign(context=lambda x: format_docs(x["context"]))
        | ANSWER_PROMPT
        | llm
        | StrOutputParser()
    )

    print("✅ Enhanced RAG Chain ready")
    return rag_chain

# ================================================================
# LINE Bot + Flask (unchanged)
# ================================================================
app = Flask(__name__)

configuration = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

rag_chain = initialize_rag_chain("micro_rag_optimized.md")

@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers.get('X-Line-Signature', "")
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return "OK"

def safe_reply(api: MessagingApi, reply_token: str, text: str):
    api.reply_message_with_http_info(
        ReplyMessageRequest(
            reply_token=reply_token,
            messages=[TextMessage(text=text[:4900])]
        )
    )

# Simple chat responses for casual questions
def get_simple_response(question: str) -> str:
    """Handle simple casual questions directly"""
    q_lower = question.lower().strip()

    casual_responses = {
        # Greetings
        r'(สวัสดี|hello|hi|ดี|hey)': "สวัสดีครับ! มีอะไรเกี่ยวกับ Microcontroller ที่อยากถามไหมครับ?",
        r'(ไง|เป็นไง|ยังไง)': "สบายดีครับ! มีคำถามเกี่ยวกับ Microcontroller ไหมครับ?",

        # Weather/general chat
        r'(อากาศ|ฝน|ร้อน|หนาว|อุณหภูมิ)': "ผมไม่สามารถตรวจสอบสภาพอากาศได้ครับ แต่ถ้าอยากรู้เรื่อง Microcontroller มีอะไรถามได้เลยนะครับ!",
        r'(กิน|อาหาร|หิว|อร่อย)': "ผมไม่สามารถแนะนำอาหารได้ครับ แต่ถ้าอยากรู้เรื่อง Microcontroller สามารถถามได้เลย!",

        # Simple questions
        r'(ทำไม|เพราะไร|why)': "ถ้าเป็นคำถามเกี่ยวกับ Microcontroller สามารถถามได้เลยครับ!",
        r'(ขอบคุณ|thanks|thank you)': "ยินดีครับ! มีอะไรเกี่ยวกับ Microcontroller อยากถามเพิ่มเติมไหมครับ?",

        # Numbers or short queries
        r'^[0-9\s\.]+$': "ผมเป็นผู้ช่วยเรื่อง Microcontroller ครับ มีคำถามเกี่ยวกับ Microcontroller ไหมครับ?",
    }

    for pattern, response in casual_responses.items():
        if re.search(pattern, q_lower):
            return response

    return None


@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):
    q_raw = event.message.text

    # Check for simple casual questions first
    simple_ans = get_simple_response(q_raw)
    if simple_ans:
        print(f"💬 Q: {q_raw} -> [Simple Response]")
        print(f"🤖 Ans: {simple_ans}")
        with ApiClient(configuration) as api_client:
            safe_reply(MessagingApi(api_client), event.reply_token, simple_ans)
        return

    # Process with RAG for technical questions
    q_processed = preprocess_question(q_raw)
    print(f"💬 Q: {q_raw} -> {q_processed}")

    try:
        ans = rag_chain.invoke(q_processed)
        # Additional fallback if no relevant answer found
        if "ไม่พบข้อมูลในเอกสาร" in ans and len(q_raw) > 10:
            # Try with original question
            ans_fallback = rag_chain.invoke(q_raw)
            if "ไม่พบข้อมูลในเอกสาร" not in ans_fallback:
                ans = ans_fallback
    except Exception as e:
        ans = f"เกิดข้อผิดพลาด: {e}"

    print(f"🤖 Ans: {ans}")
    with ApiClient(configuration) as api_client:
        safe_reply(MessagingApi(api_client), event.reply_token, ans)

# ================================================================
# Run server + Ngrok (unchanged)
# ================================================================
if __name__ == "__main__":
    ngrok.set_auth_token(NGROK_AUTHTOKEN)

    # Disconnect any existing tunnels
    try:
        tunnels = ngrok.get_tunnels()
        for tunnel in tunnels:
            ngrok.disconnect(tunnel.public_url)
    except Exception as e:
        print(f"Could not disconnect existing ngrok tunnels: {e}")

    public_url = ngrok.connect(5000)
    print("="*60)
    print("🚀 Enhanced LINE Bot Online")
    print(f"🔗 Webhook URL: {public_url.public_url}/callback")
    print("="*60)
    app.run(port=5000, debug=False)