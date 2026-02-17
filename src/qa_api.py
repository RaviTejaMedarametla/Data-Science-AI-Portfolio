from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, TokenTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma.vectorstores import Chroma


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("qa_api")


class QARequest(BaseModel):
    question_lecture: str = Field(..., min_length=1)
    question_title: str = Field(..., min_length=1)
    question_body: str = Field(..., min_length=1)


class QAResponse(BaseModel):
    answer: str
    confidence: float
    citations: List[str]
    latency_ms: float
    retrieval_accuracy: float
    hallucination_flag: bool


class HealthResponse(BaseModel):
    ready: bool


class MonitoringResponse(BaseModel):
    requests_total: int
    avg_latency_ms: float
    avg_retrieval_accuracy: float
    hallucination_rate: float


@asynccontextmanager
async def lifespan(_: FastAPI):
    startup()
    yield


app = FastAPI(title="LangChain QA API", version="1.1.0", lifespan=lifespan)

llm: Optional[ChatOpenAI] = None
retriever = None
chat_prompt_template_retrieving: Optional[ChatPromptTemplate] = None

monitoring: Dict[str, Any] = {
    "requests_total": 0,
    "latency_ms_total": 0.0,
    "retrieval_accuracy_total": 0.0,
    "hallucination_count": 0,
}


PROMPT_RETRIEVING_S = """You are a helpful teaching assistant for a Tableau course.
You will receive a student question and supporting context passages.

Rules:
1) Answer ONLY using the supplied context.
2) If context is insufficient, say exactly: "I don't have enough context to answer confidently."
3) Add a short "Citations" section at the end.
4) Each citation must use this format:
   - [Section: <section>, Lecture: <lecture>]
5) Do not invent citations.
"""


def _build_question(req: QARequest) -> str:
    return f"Lecture: {req.question_lecture}\nTitle: {req.question_title}\nBody: {req.question_body}"


def _format_context(docs: List[Document]) -> str:
    chunks = []
    for i, doc in enumerate(docs, start=1):
        section = doc.metadata.get("section", "Unknown Section")
        lecture = doc.metadata.get("lecture", "Unknown Lecture")
        chunks.append(f"[{i}] Section: {section} | Lecture: {lecture}\n{doc.page_content.strip()}")
    return "\n\n".join(chunks)


def _extract_citations(answer_text: str) -> List[str]:
    pattern = r"\[Section:\s*.*?,\s*Lecture:\s*.*?\]"
    return re.findall(pattern, answer_text)


def _allowed_citations_from_docs(docs: List[Document]) -> set[str]:
    allowed = set()
    for doc in docs:
        section = str(doc.metadata.get("section", "Unknown Section")).strip()
        lecture = str(doc.metadata.get("lecture", "Unknown Lecture")).strip()
        allowed.add(f"[Section: {section}, Lecture: {lecture}]")
    return allowed


def _compute_retrieval_accuracy(citations: List[str], docs: List[Document]) -> float:
    if not citations:
        return 0.0
    allowed = _allowed_citations_from_docs(docs)
    valid = sum(1 for c in citations if c in allowed)
    return round(valid / len(citations), 4)


def _compute_confidence(answer_text: str, retrieved_docs: List[Document], retrieval_accuracy: float) -> float:
    if not retrieved_docs:
        return 0.05
    coverage = min(len(retrieved_docs) / 4.0, 1.0)
    nonempty = 1.0 if len(answer_text.strip()) > 20 else 0.0
    return round(0.4 * coverage + 0.4 * retrieval_accuracy + 0.2 * nonempty, 4)


def _update_monitoring(latency_ms: float, retrieval_accuracy: float, hallucination_flag: bool) -> None:
    monitoring["requests_total"] += 1
    monitoring["latency_ms_total"] += latency_ms
    monitoring["retrieval_accuracy_total"] += retrieval_accuracy
    if hallucination_flag:
        monitoring["hallucination_count"] += 1


async def _ainvoke_answer(question: str, context: str) -> str:
    if llm is None or chat_prompt_template_retrieving is None:
        raise HTTPException(status_code=503, detail="LLM is not configured.")
    messages = chat_prompt_template_retrieving.format_messages(question=question, context=context)
    result = await llm.ainvoke(messages)
    return result.content if hasattr(result, "content") else str(result)


def startup() -> None:
    global llm, retriever, chat_prompt_template_retrieving

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.info("OPENAI_API_KEY is not set. QA endpoints will return fallback responses.")
        return

    transcript_path = Path(os.getenv("QA_TRANSCRIPT_PDF", "tableau_course_transcript.pdf"))
    model_name = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    chroma_collection = os.getenv("QA_CHROMA_COLLECTION", "tableau_qa_collection")

    if transcript_path.exists():
        docs = PyPDFLoader(str(transcript_path)).load()
        transcript_text = "\n\n".join(d.page_content for d in docs)
    else:
        logger.warning("Transcript PDF not found at %s. Using fallback sample.", transcript_path)
        transcript_text = (
            "# Section: Calculations\n"
            "## Lecture: Adding a custom calculation\n"
            "In this lecture we explain why SUM is used in GM% calculations.\n"
            "# Section: Visual Analytics\n"
            "## Lecture: Building charts\n"
            "We compare bar and line charts for trend analysis."
        )

    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "section"), ("##", "lecture"), ("###", "topic")],
        strip_headers=False,
    )
    md_docs = md_splitter.split_text(transcript_text)

    token_splitter = TokenTextSplitter(chunk_size=350, chunk_overlap=50)
    chunks = token_splitter.split_documents(md_docs)

    embeddings = OpenAIEmbeddings(model=embedding_model)
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=chroma_collection,
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    llm = ChatOpenAI(model=model_name, temperature=0)

    chat_prompt_template_retrieving = ChatPromptTemplate.from_messages(
        [
            ("system", PROMPT_RETRIEVING_S),
            ("human", "Question:\n{question}\n\nContext:\n{context}"),
        ]
    )

    logger.info("QA API initialized | model=%s | chunks=%d", model_name, len(chunks))


@app.get('/health', response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(ready=bool(retriever is not None and llm is not None))


@app.get('/monitoring', response_model=MonitoringResponse)
async def monitoring_metrics() -> MonitoringResponse:
    total = max(monitoring['requests_total'], 1)
    return MonitoringResponse(
        requests_total=monitoring['requests_total'],
        avg_latency_ms=round(monitoring['latency_ms_total'] / total, 4),
        avg_retrieval_accuracy=round(monitoring['retrieval_accuracy_total'] / total, 4),
        hallucination_rate=round(monitoring['hallucination_count'] / total, 4),
    )


@app.post("/qa", response_model=QAResponse)
async def qa_answer(payload: QARequest) -> QAResponse:
    start = time.perf_counter()

    if retriever is None or llm is None:
        fallback = "I don't have enough context to answer confidently."
        latency_ms = round((time.perf_counter() - start) * 1000, 4)
        _update_monitoring(latency_ms, 0.0, False)
        return QAResponse(
            answer=fallback,
            confidence=0.0,
            citations=[],
            latency_ms=latency_ms,
            retrieval_accuracy=0.0,
            hallucination_flag=False,
        )

    question = _build_question(payload)
    docs = await retriever.ainvoke(question)

    if not docs:
        fallback = "I don't have enough context to answer confidently."
        latency_ms = round((time.perf_counter() - start) * 1000, 4)
        _update_monitoring(latency_ms, 0.0, False)
        return QAResponse(
            answer=fallback,
            confidence=0.05,
            citations=[],
            latency_ms=latency_ms,
            retrieval_accuracy=0.0,
            hallucination_flag=False,
        )

    context = _format_context(docs)
    answer = await _ainvoke_answer(question, context)
    citations = _extract_citations(answer)
    retrieval_accuracy = _compute_retrieval_accuracy(citations, docs)
    hallucination_flag = bool(citations) and retrieval_accuracy < 1.0
    confidence = _compute_confidence(answer, docs, retrieval_accuracy)
    latency_ms = round((time.perf_counter() - start) * 1000, 4)

    _update_monitoring(latency_ms, retrieval_accuracy, hallucination_flag)

    logger.info(
        "qa_request_complete | latency_ms=%.4f retrieval_accuracy=%.4f hallucination=%s",
        latency_ms,
        retrieval_accuracy,
        hallucination_flag,
    )

    return QAResponse(
        answer=answer,
        confidence=confidence,
        citations=citations,
        latency_ms=latency_ms,
        retrieval_accuracy=retrieval_accuracy,
        hallucination_flag=hallucination_flag,
    )


async def _stream_tokens(payload: QARequest) -> AsyncGenerator[bytes, None]:
    start = time.perf_counter()

    if retriever is None or llm is None or chat_prompt_template_retrieving is None:
        fallback = {"token": "I don't have enough context to answer confidently."}
        yield f"data: {json.dumps(fallback)}\n\n".encode("utf-8")
        return

    question = _build_question(payload)
    docs = await retriever.ainvoke(question)

    if not docs:
        fallback = {"token": "I don't have enough context to answer confidently."}
        yield f"data: {json.dumps(fallback)}\n\n".encode("utf-8")
        return

    context = _format_context(docs)
    messages = chat_prompt_template_retrieving.format_messages(question=question, context=context)

    full_text = []
    async for chunk in llm.astream(messages):
        token = chunk.content if hasattr(chunk, "content") else str(chunk)
        if token:
            full_text.append(token)
            yield f"data: {json.dumps({'token': token})}\n\n".encode("utf-8")
            await asyncio.sleep(0)

    answer_text = "".join(full_text)
    citations = _extract_citations(answer_text)
    retrieval_accuracy = _compute_retrieval_accuracy(citations, docs)
    hallucination_flag = bool(citations) and retrieval_accuracy < 1.0
    confidence = _compute_confidence(answer_text, docs, retrieval_accuracy)
    latency_ms = round((time.perf_counter() - start) * 1000, 4)

    _update_monitoring(latency_ms, retrieval_accuracy, hallucination_flag)

    final_payload = {
        "done": True,
        "confidence": confidence,
        "citations": citations,
        "latency_ms": latency_ms,
        "retrieval_accuracy": retrieval_accuracy,
        "hallucination_flag": hallucination_flag,
    }
    yield f"data: {json.dumps(final_payload)}\n\n".encode("utf-8")


@app.post("/qa/stream")
async def qa_answer_stream(payload: QARequest) -> StreamingResponse:
    return StreamingResponse(_stream_tokens(payload), media_type="text/event-stream")
