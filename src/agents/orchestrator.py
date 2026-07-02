"""
Orchestrator — entry point for the Legal RAG system.
Receives a user question, routes to the agent, and returns the result.
"""
from __future__ import annotations
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def answer_question(question: str) -> str:
    """
    Main entry point: takes a user question, runs the agent, and returns an answer.
    """
    from src.agents.rag_agent import ask
    return ask(question)


def chat(question: str, history: list[dict] | None = None) -> str:
    """
    Chat-compatible entry point for Gradio.
    Accepts question + conversation history and returns answer string.
    """
    # For now, each question is treated independently (stateless).
    # History can be used in the future for multi-turn conversations.
    return answer_question(question)
