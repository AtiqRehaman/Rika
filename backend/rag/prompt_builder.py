# rag/prompt_builder.py

import tiktoken
from typing import List, Dict, Any

class PromptBuilder:
    """
    Builds prompts that fit within the model's context window.
    Trims retrieved context intelligently rather than blindly truncating.
    """

    # Leave room for response generation
    MAX_CONTEXT_TOKENS = 2048
    MAX_RESPONSE_TOKENS = 512

    def __init__(self):
        # cl100k is close enough for most GGUF models
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def build(self, query: str, retrieved: List[Dict[str, Any]]) -> str:
        system = (
            "You are an expert coding assistant specializing in TensorFlow and PyTorch. "
            "Answer questions using ONLY the provided context. "
            "If the context doesn't contain the answer, say so clearly. "
            "Always include working code examples when relevant."
        )

        context_parts = []
        used_tokens = self._count_tokens(system) + self._count_tokens(query) + 100  # buffer

        for i, chunk in enumerate(retrieved):
            source = chunk["metadata"].get("source", "unknown")
            snippet = f"[Source {i+1}: {source}]\n{chunk['content']}"
            snippet_tokens = self._count_tokens(snippet)

            if used_tokens + snippet_tokens > self.MAX_CONTEXT_TOKENS:
                break  # stop adding context rather than overflow

            context_parts.append(snippet)
            used_tokens += snippet_tokens

        context_block = "\n\n---\n\n".join(context_parts) if context_parts else "No relevant context found."

        prompt = f"""<|system|>
{system}

### Retrieved Context:
{context_block}
<|end|>

<|user|>
{query}
<|end|>

<|assistant|>"""

        return prompt