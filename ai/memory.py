"""
Conversation memory: add turns (human, assistant) and get history for context.

Uses an in-memory buffer of LangChain messages. Expose add_turn, get_history,
and get_messages_for_context for use in the chat/RAG flow.
Run from project root: python ai/memory.py
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


class ConversationMemory:
    """
    In-memory conversation buffer. Stores (human, assistant) turns as
    LangChain messages and can format them for inclusion in prompts.
    """

    def __init__(self) -> None:
        self._messages: list[BaseMessage] = []

    def add_turn(self, human: str, assistant: str) -> None:
        """Append one conversation turn: user said `human`, assistant said `assistant`."""
        self._messages.append(HumanMessage(content=human))
        self._messages.append(AIMessage(content=assistant))

    def get_history(self) -> list[BaseMessage]:
        """Return the list of messages (HumanMessage, AIMessage) in order."""
        return list(self._messages)

    def get_message_count(self) -> int:
        """Return the number of messages in the buffer."""
        return len(self._messages)

    def get_messages_for_context(self, max_turns: int | None = None) -> str:
        """
        Format recent turns as a string for inclusion in a prompt (e.g. prior context for RAG).

        Args:
            max_turns: If set, include only the last N turns (each turn = one human + one assistant).
                       If None, include all turns.
        """
        messages = self._messages
        if max_turns is not None and max_turns > 0:
            # Each turn = 2 messages (human, assistant)
            keep = max_turns * 2
            messages = messages[-keep:] if len(messages) > keep else messages
        lines: list[str] = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                lines.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                lines.append(f"Assistant: {msg.content}")
        return "\n".join(lines) if lines else ""

    def clear(self) -> None:
        """Remove all messages from the buffer."""
        self._messages.clear()


if __name__ == "__main__":
    # (1) Add two turns (human + assistant), get history, print message count or last message.
    mem = ConversationMemory()
    mem.add_turn("What were sales by region?", "East: 320,296; North: 353,025; South: 348,516; West: 361,383.")
    mem.add_turn("And for Q3?", "Here are Q3 regional totals: East 82,000; North 89,000; South 87,000; West 91,000.")
    history = mem.get_history()
    print("After two turns:")
    print("Message count:", mem.get_message_count())
    print("Last message (content preview):", (history[-1].content or "")[:80] + "...")
    print("Context string (first 200 chars):", mem.get_messages_for_context()[:200] + "...")
    print()

    # (2) Clear or reset and add one turn, print again.
    mem.clear()
    mem.add_turn("Which product sold the most?", "Widget A had the highest total sales.")
    history2 = mem.get_history()
    print("After clear and one turn:")
    print("Message count:", mem.get_message_count())
    print("Last message:", (history2[-1].content or "").strip())
