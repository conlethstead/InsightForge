"""
RAG pipeline: question → retriever → context → prompt → LLM → answer.

Run from project root: python ai/rag.py
"""

import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from langchain_core.documents import Document

from ai.prompts import get_rag_user_prompt, get_system_prompt
from retrieval.retriever import retrieve

logger = logging.getLogger("insightforge.rag")


def _docs_to_context(docs: list) -> str:
    """Turn a list of LangChain Documents into a single context string."""
    parts = [doc.page_content for doc in docs if getattr(doc, "page_content", None)]
    return "\n\n---\n\n".join(parts) if parts else ""


def rag_answer(
    question: str,
    top_k: int = 10,
    conversation_context: str | None = None,
    return_context: bool = False,
    chunks: list[Document] | None = None,
) -> str | tuple[str, str]:
    """
    Run RAG: retrieve context for the question, then call LLM to produce an answer.

    Args:
        question: User question.
        top_k: Max number of chunks passed to the retriever.
        conversation_context: Optional prior conversation string for follow-ups.
        return_context: If True, return (answer, context_used) for debugging.
        chunks: Optional prebuilt chunks (e.g. session_state.chunks). If None, retriever uses get_chunks().

    Returns:
        LLM answer string, or (answer, context) if return_context=True.
        Raises ValueError if API key is not set.
    """
    if not config.has_api_key():
        raise ValueError("PAID_OPENAI_API_KEY is not set. Set it in .zshrc or .env.")

    # Step 1: RAG input
    conv_len = len(conversation_context or "")
    logger.info("[RAG] step=input question=%r top_k=%d chunks_provided=%s conversation_context_len=%d", question, top_k, chunks is not None, conv_len)

    # Step 2: Retrieve
    docs = retrieve(question, chunks=chunks, top_k=top_k)
    logger.info("[RAG] step=retrieve num_docs=%d", len(docs))
    for i, doc in enumerate(docs[:3]):
        preview = (doc.page_content[:80] + "…") if len(doc.page_content) > 80 else doc.page_content
        logger.info("[RAG] step=retrieve doc[%d] preview=%s", i, preview)
    if len(docs) > 3:
        logger.info("[RAG] step=retrieve ... and %d more docs", len(docs) - 3)

    # Step 3: Build context
    context = _docs_to_context(docs)
    logger.info("[RAG] step=context context_len=%d", len(context))
    logger.info("[RAG] step=context context_preview=%s", (context[:500] + "…" if len(context) > 500 else context))

    llm = ChatOpenAI(
        api_key=config.PAID_OPENAI_API_KEY,
        model="gpt-4o-mini",
        temperature=0,
    )
    system_prompt = get_system_prompt()
    user_prompt = get_rag_user_prompt(context, question, conversation_context)
    # Step 4: Prompts
    logger.info("[RAG] step=prompts system_len=%d user_len=%d", len(system_prompt), len(user_prompt))
    logger.debug("[RAG] step=prompts user_prompt_full=%s", user_prompt)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    # Step 5: LLM call
    logger.info("[RAG] step=llm model=gpt-4o-mini invoking")
    response = llm.invoke(messages)
    answer = (response.content or "").strip()
    # Step 6: RAG output
    logger.info("[RAG] step=output answer_len=%d", len(answer))
    logger.info("[RAG] step=output answer_preview=%s", (answer[:300] + "…" if len(answer) > 300 else answer))
    if return_context:
        return answer, context
    return answer


if __name__ == "__main__":
    # (1) Call RAG with one fixed question, print answer length or first 300 chars.
    q1 = "What were sales by region?"
    print("Question 1:", q1)
    try:
        ans1 = rag_answer(q1)
        print("Answer length:", len(ans1))
        print("First 300 chars:", ans1[:300] + ("..." if len(ans1) > 300 else ""))
    except ValueError as e:
        print("Error:", e)
    except Exception as e:
        print("Error:", e)
    print()

    # (2) Call with a second question, print that answer.
    q2 = "Which products sold the most?"
    print("Question 2:", q2)
    try:
        ans2 = rag_answer(q2)
        print("Answer length:", len(ans2))
        print("First 300 chars:", ans2[:300] + ("..." if len(ans2) > 300 else ""))
    except ValueError as e:
        print("Error:", e)
    except Exception as e:
        print("Error:", e)
