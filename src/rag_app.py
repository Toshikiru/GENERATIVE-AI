"""
Checkpoint 3 - RAG Application (LangChain)

SyllaBot's main RAG application: retrieves relevant syllabus context from
the Chroma vector store and grounds Gemini responses in it, with
multi-turn conversational memory so follow-up questions ("what about the
midterm?") resolve using prior turns.

Run modes:
    python src/rag_app.py            interactive CLI chat
    python src/rag_app.py --demo     runs a scripted 6-question demo
                                      (5 grounded + 1 out-of-scope refusal)
                                      and prints the full transcript
"""
import os
import sys
from pathlib import Path

from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = ROOT / "data" / "chroma_db"
COLLECTION_NAME = "syllabot_chunks"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-3.5-flash"
# Query-condensing (resolving "it"/"that" in follow-ups before retrieval)
# runs on every conversational turn, doubling calls against whichever model
# handles it. Using a lighter/higher-free-tier-quota model for that
# auxiliary step keeps the strict daily quota on GEMINI_MODEL free for the
# answers that actually matter.
CONDENSE_MODEL = "gemini-3.5-flash"
TOP_K = 3

SYSTEM_PROMPT = """You are SyllaBot, a course syllabus and study guide assistant.

You will be given retrieved excerpts from the student's course syllabi.
Answer using ONLY the information in the provided excerpts, and use the
conversation history to resolve follow-up questions (e.g. "what about the
midterm?" referring to a course discussed earlier).

Rules:
1. If the excerpts contain the answer, respond concisely and cite which
   course/document it came from.
2. If the excerpts do NOT contain enough information to answer, say so
   explicitly: "I couldn't find that in your uploaded syllabi." Do not
   guess or fill in with general knowledge about how courses "usually" work.
3. Never invent a percentage, date, or policy not literally present in the
   excerpts.
4. If the question is unrelated to course syllabi, politely decline.

Retrieved excerpts:
{context}"""

DEMO_CONVERSATION = [
    "How much is the final exam worth in the microeconomics course?",
    "What textbook does it recommend if I want to read further?",
    "What programming language is used in the data science course?",
    "Can students collaborate on problem sets in the database systems course?",
    "What's the late policy for the Python programming course?",
    "What is the capital of France?",  # deliberately out-of-scope
]


class SentenceTransformerEmbeddings(Embeddings):
    """Wraps our Checkpoint 1 embedding model to the LangChain Embeddings
    interface, so the Chroma retriever stays consistent with the model
    justified in Checkpoint 1 instead of defaulting to a different one."""

    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text):
        return self.model.encode(text, convert_to_numpy=True).tolist()


def extract_text(content) -> str:
    """
    Gemini via langchain_google_genai can return .content as a plain string
    OR as a list of content blocks (text blocks plus internal 'thought
    signature' blocks used for multi-turn reasoning continuity). Pull out
    just the human-readable text either way.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            block.get("text", "") for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "".join(parts).strip()
    return str(content)


CONDENSE_PROMPT = """Given the conversation history and a follow-up question, \
rewrite the follow-up as a standalone question that includes any context \
needed to understand it on its own (e.g. which course is being discussed).

Only add context for genuinely vague references: bare pronouns ("it", \
"that") or bare phrases with no identifying detail ("the course", "that \
class"). If the follow-up already names or describes a specific \
course/subject/topic (e.g. "the Python programming course", "the database \
course"), you MUST keep that description exactly as written -- do NOT \
substitute a different course from the history, even if it was discussed \
more recently or seems topically related (e.g. don't assume "the Python \
programming course" means "the data science course" just because a data \
science course was said to use Python).

If the follow-up is already standalone, return it unchanged. Return ONLY \
the rewritten question, nothing else.

Conversation history:
{history}

Follow-up question: {question}

Standalone question:"""


def build_chain():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY is not set.")
        print('  PowerShell:  $env:GEMINI_API_KEY = "your-key-here"')
        sys.exit(1)

    embeddings = SentenceTransformerEmbeddings(EMBED_MODEL_NAME)
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    # Hybrid retrieval: dense/semantic search alone can't tell apart two
    # courses' near-identical boilerplate ("Collaboration Policy" reads
    # almost the same in every syllabus), so a query naming a specific
    # course ("...in the database systems course") can lose to a
    # semantically-similar chunk from a different course. BM25 (exact
    # keyword overlap) reliably catches distinctive terms like "database
    # systems" that dense embeddings under-weight, so combining both
    # retrieval methods catches cases either one misses alone.
    all_docs = vectorstore.get(include=["documents", "metadatas"])
    from langchain_core.documents import Document
    bm25_corpus = [
        Document(page_content=doc, metadata=meta)
        for doc, meta in zip(all_docs["documents"], all_docs["metadatas"])
    ]
    bm25_retriever = BM25Retriever.from_documents(bm25_corpus)
    bm25_retriever.k = TOP_K

    retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever], weights=[0.5, 0.5]
    )

    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, google_api_key=api_key)
    condense_llm = ChatGoogleGenerativeAI(model=CONDENSE_MODEL, google_api_key=api_key)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ])

    def condense_question(question: str, history: list) -> str:
        """
        Retrieval runs on the raw question text BEFORE the main LLM call
        ever sees it, so a follow-up like "what textbook does it
        recommend?" would otherwise be searched literally -- "it" means
        nothing to a vector/keyword search, so retrieval can miss the
        right document entirely even though the LLM itself could have
        resolved the pronoun from history. Resolving it into a standalone
        question first (a small extra LLM call) fixes retrieval, not just
        generation.
        """
        if not history:
            return question
        history_text = "\n".join(
            f"{'Student' if m.type == 'human' else 'SyllaBot'}: {extract_text(m.content)}"
            for m in history
        )
        result = condense_llm.invoke(CONDENSE_PROMPT.format(history=history_text, question=question))
        return extract_text(result.content).strip() or question

    def retrieve_context(inputs):
        question = condense_question(inputs["question"], inputs.get("history", []))
        docs = retriever.invoke(question)
        if not docs:
            return "(no matching syllabus content found)"
        return "\n\n".join(f"[{d.metadata['source']}]\n{d.page_content}" for d in docs)

    chain = (
        RunnablePassthrough.assign(context=retrieve_context)
        | prompt
        | llm
    )

    session_store = {}

    def get_session_history(session_id: str):
        if session_id not in session_store:
            session_store[session_id] = InMemoryChatMessageHistory()
        return session_store[session_id]

    return RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )


def run_demo(app):
    config = {"configurable": {"session_id": "demo-session"}}
    print("=" * 70)
    print("SCRIPTED DEMO: 5 grounded queries + 1 out-of-scope refusal")
    print("=" * 70)
    for question in DEMO_CONVERSATION:
        response = app.invoke({"question": question}, config=config)
        print(f"\nStudent: {question}")
        print(f"SyllaBot: {extract_text(response.content)}")


def run_interactive(app):
    config = {"configurable": {"session_id": "interactive-session"}}
    print("SyllaBot RAG demo. Type 'quit' to exit.\n")
    while True:
        question = input("Student: ").strip()
        if question.lower() in ("quit", "exit"):
            break
        if not question:
            continue
        response = app.invoke({"question": question}, config=config)
        print(f"SyllaBot: {extract_text(response.content)}\n")


if __name__ == "__main__":
    rag_chain = build_chain()
    if "--demo" in sys.argv:
        run_demo(rag_chain)
    else:
        run_interactive(rag_chain)
