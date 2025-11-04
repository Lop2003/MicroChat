# This file is a copy of `code.py` but renamed to avoid shadowing the stdlib `code` module.
# You can run `python app.py` instead of `python code.py` when using VS Code or a local env.

import os
from dotenv import load_dotenv

# Load environment variables from .env when running locally
load_dotenv()
import re
import sys
import subprocess
from typing import List, Dict, Any
from collections import defaultdict, deque
from datetime import datetime

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
"""from pyngrok import ngrok"""

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

from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# ================================================================
# Conversation Memory Store
# ================================================================
# Store conversation history per user (LINE user_id)
# Format: {user_id: deque([(question, answer, timestamp), ...], maxlen=10)}
conversation_memory = defaultdict(lambda: deque(maxlen=10))

def add_to_memory(user_id: str, question: str, answer: str):
    """Add a Q&A pair to user's conversation memory"""
    conversation_memory[user_id].append({
        'question': question,
        'answer': answer,
        'timestamp': datetime.now().isoformat()
    })

def get_conversation_context(user_id: str, max_turns: int = 5) -> str:
    """Get recent conversation history as formatted string"""
    history = list(conversation_memory[user_id])
    if not history:
        return "ไม่มีประวัติการสนทนา"
    
    # Get last N turns
    recent = history[-max_turns:]
    formatted = []
    for i, turn in enumerate(recent, 1):
        formatted.append(f"Q{i}: {turn['question']}")
        formatted.append(f"A{i}: {turn['answer']}")
    
    return "\n".join(formatted)

def clear_user_memory(user_id: str):
    """Clear conversation history for a user"""
    if user_id in conversation_memory:
        conversation_memory[user_id].clear()

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
# Enhanced Query Preprocessing with Context Awareness
# ================================================================
def preprocess_question(q: str, conversation_context: str = "") -> str:
    """Enhanced query preprocessing for better retrieval with context awareness"""
    low = q.lower()

    # Handle follow-up questions that reference previous context
    follow_up_patterns = [
        r'(แล้ว|ล่ะ|อีก|เพิ่มเติม|ต่อ)',
        r'(นั่น|นี่|มัน|ตัวนั้น|อันนั้น)',
        r'(อย่างไร|ยังไง|อะไรบ้าง)',
    ]
    
    is_follow_up = any(re.search(pattern, low) for pattern in follow_up_patterns)
    
    # If it's a follow-up and we have context, include it
    context_hint = ""
    if is_follow_up and conversation_context:
        # Extract key terms from recent conversation
        context_hint = f" [บริบท: {conversation_context[:200]}]"

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

    result = original_query
    if expansions:
        result = f"{original_query} {' '.join(expansions)}"
    
    # Add context hint if available
    if context_hint:
        result = f"{result}{context_hint}"
    
    return result

# ================================================================
# Initialize Enhanced RAG chain
# ================================================================
def initialize_rag_chain(md_file: str = "dataset.md"):
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

    # Improved prompt template with conversation awareness
    template = """คุณคือผู้ช่วยผู้เชี่ยวชาญด้านไมโครคอนโทรลเลอร์ที่เป็นมิตรและช่วยเหลือดี

กฎการตอบ:
1. ใช้ข้อมูลจาก Context และประวัติการสนทนา (ถ้ามี) เพื่อตอบคำถาม
2. ตอบด้วยภาษาที่เป็นธรรมชาติ เข้าใจง่าย และเป็นกันเอง
3. หากคำถามอ้างอิงถึงคำตอบก่อนหน้า (เช่น "แล้วมันคืออะไร" "อย่างไร" "ต่อ") ให้ใช้บริบทจากประวัติการสนทนา
4. ห้ามใส่เครื่องหมาย * หรือ markdown formatting
5. ห้ามเขียน "จากเอกสารที่ให้มา" หรือ "(ข้อมูลที่ X)"
6. ใช้รูปแบบย่อหน้าธรรมดา หรือขึ้นบรรทัดใหม่ด้วย - เท่านั้น
7. หากไม่พบข้อมูลให้ตอบ "ไม่พบข้อมูลในเอกสาร แต่คุณสามารถถามคำถามอื่นได้ครับ"
8. ตอบให้ครบถ้วนและละเอียด แต่กระชับ

ประวัติการสนทนา:
{conversation_history}

Context จากเอกสาร:
{context}

คำถามปัจจุบัน: {question}

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

    def create_rag_chain_with_history(conversation_history: str = "ไม่มีประวัติการสนทนา"):
        """Create RAG chain with conversation history"""
        return (
            {
                "context": retrieval,
                "question": RunnablePassthrough(),
                "conversation_history": lambda x: conversation_history,
            }
            | RunnablePassthrough.assign(context=lambda x: format_docs(x["context"]))
            | ANSWER_PROMPT
            | llm
            | StrOutputParser()
        )

    print("✅ Enhanced RAG Chain with Memory ready")
    return create_rag_chain_with_history

# ================================================================
# LINE Bot + Flask (unchanged)
# ================================================================
app = Flask(__name__)

configuration = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

rag_chain_factory = initialize_rag_chain("dataset.md")

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

# Enhanced simple response handling to avoid mismatches and ensure technical questions are processed

def get_simple_response(question: str, user_id: str = None) -> str:
    """Handle simple casual questions directly"""
    q_lower = question.lower().strip()

    # Check for memory clear command
    if re.search(r'(ลบประวัติ|ล้างประวัติ|clear|reset|เริ่มใหม่)', q_lower):
        if user_id:
            clear_user_memory(user_id)
            return "ล้างประวัติการสนทนาเรียบร้อยแล้วครับ! เริ่มคุยกันใหม่ได้เลย"

    # Enhanced casual responses with better matching
    casual_responses = {
        # Greetings
        r'^(สวัสดี|hello|hi|ดี|hey)$': "สวัสดีครับ! ผมเป็นผู้ช่วยด้าน Microcontroller พร้อมตอบคำถามของคุณครับ มีอะไรอยากรู้ไหมครับ?",
        r'^(ไง|เป็นไง|ยังไง)$': "สบายดีครับ! มีคำถามเกี่ยวกับ Microcontroller ไหมครับ?",

        # Weather/general chat
        r'(อากาศ|ฝน|ร้อน|หนาว|อุณหภูมิ)': "ผมไม่สามารถตรวจสอบสภาพอากาศได้ครับ แต่ถ้าอยากรู้เรื่อง Microcontroller มีอะไรถามได้เลยนะครับ!",
        r'(กิน|อาหาร|หิว|อร่อย)': "ผมไม่สามารถแนะนำอาหารได้ครับ แต่ถ้าอยากรู้เรื่อง Microcontroller สามารถถามได้เลย!",

        # Simple questions
        r'^(ทำไม|เพราะไร|why)$': "ถ้าเป็นคำถามเกี่ยวกับ Microcontroller สามารถถามได้เลยครับ!",
        r'(ขอบคุณ|thanks|thank you)': "ยินดีครับ! มีอะไรเกี่ยวกับ Microcontroller อยากถามเพิ่มเติมไหมครับ?",

        # Numbers or short queries
        r'^[0-9\s\.]+$': "ผมเป็นผู้ช่วยเรื่อง Microcontroller ครับ มีคำถามเกี่ยวกับ Microcontroller ไหมครับ?",
    }

    # Check for casual responses
    for pattern, response in casual_responses.items():
        if re.search(pattern, q_lower):
            return response

    # If no match, return None to allow further processing
    return None


# Update message handler with conversation memory
@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):
    q_raw = event.message.text
    user_id = event.source.user_id  # Get LINE user ID for memory

    # Check for simple casual questions first
    simple_ans = get_simple_response(q_raw, user_id)
    if simple_ans:
        print(f"💬 User {user_id[:8]}... Q: {q_raw} -> [Simple Response]")
        print(f"🤖 Ans: {simple_ans}")
        with ApiClient(configuration) as api_client:
            safe_reply(MessagingApi(api_client), event.reply_token, simple_ans)
        # Don't save casual responses to memory
        return

    # Get conversation history for context
    conv_context = get_conversation_context(user_id, max_turns=3)
    
    # Process with RAG for technical questions
    q_processed = preprocess_question(q_raw, conv_context)
    print(f"💬 User {user_id[:8]}... Q: {q_raw}")
    print(f"🔍 Processed: {q_processed[:150]}...")
    print(f"📚 Context turns: {len(conversation_memory[user_id])}")

    try:
        # Create RAG chain with current conversation history
        rag_chain = rag_chain_factory(conv_context)
        ans = rag_chain.invoke(q_processed)
        
        # Additional fallback if no relevant answer found
        if "ไม่พบข้อมูลในเอกสาร" in ans and len(q_raw) > 10:
            # Try with original question
            rag_chain_fallback = rag_chain_factory(conv_context)
            ans_fallback = rag_chain_fallback.invoke(q_raw)
            if "ไม่พบข้อมูลในเอกสาร" not in ans_fallback:
                ans = ans_fallback
    except Exception as e:
        print(f"❌ Error: {e}")
        ans = f"เกิดข้อผิดพลาด: {e}"

    # Ensure fallback for unhandled cases
    if not ans or len(ans.strip()) == 0:
        ans = "ไม่พบข้อมูลในเอกสาร แต่คุณสามารถถามคำถามอื่นได้ครับ"

    print(f"🤖 Ans: {ans[:200]}...")
    
    # Save to conversation memory
    add_to_memory(user_id, q_raw, ans)
    
    with ApiClient(configuration) as api_client:
        safe_reply(MessagingApi(api_client), event.reply_token, ans)

# ================================================================
# Run server + Ngrok (unchanged)
# ================================================================
if __name__ == "__main__":
    # ngrok integration is commented out for Render deploy
    # ngrok.set_auth_token(NGROK_AUTHTOKEN)
    # try:
    #     tunnels = ngrok.get_tunnels()
    #     for tunnel in tunnels:
    #         ngrok.disconnect(tunnel.public_url)
    # except Exception as e:
    #     print(f"Could not disconnect existing ngrok tunnels: {e}")
    # public_url = ngrok.connect(5000)
    # print("="*60)
    # print("🚀 Enhanced LINE Bot Online")
    # print(f"🔗 Webhook URL: {public_url.public_url}/callback")
    # print("="*60)
    app.run(port=5000, debug=False)
