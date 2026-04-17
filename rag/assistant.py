from phi.agent import Agent
from phi.model.ollama import Ollama


def create_agent(model_id: str = "phi") -> Agent:
    return Agent(
        model=Ollama(id=model_id),
        instructions="""
        You are a financial AI assistant.
        Answer only from the provided context.
        Be concise and accurate.
        """
    )


def is_summary_request(question: str) -> bool:
    summary_triggers = ["summarize", "summarise", "summary", "summarizing", "summaries"]
    return any(trigger in question.lower() for trigger in summary_triggers)


def build_prompt(context: str, question: str, summary: bool = False) -> str:
    if summary:
        return f"""
You are a financial summarization assistant.
Use ONLY the context below to write a concise financial summary.
The summary should describe the key financial points from the retrieved content.
If the context is incomplete, say: \"Summary based on retrieved sections.\" Do NOT say the document is too long.
Do NOT refuse, do NOT say you cannot summarize.
Do NOT add unrelated commentary.

Context:
{context}

Question:
{question}

Summary:
"""

    return f"""
You are a financial AI assistant.

STRICT RULES:
- Answer ONLY from the context
- Do NOT guess
- Return ONLY the final answer
- No extra explanation
- If answer not found, say \"Not found\"

Context:
{context}

Question:
{question}

Final Answer:
"""
