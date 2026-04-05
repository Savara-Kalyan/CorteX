import logging
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from settings.config import settings

logger = logging.getLogger(__name__)


class QueryReformulator:

    def __init__(self):
        cfg = settings.query_understanding
        self._llm = ChatOpenAI(model=cfg.model, temperature=cfg.temperature)

    async def reformulate(self, query: str) -> str:
        try:
            chain = self._get_prompt() | self._llm | StrOutputParser()
            return (await chain.ainvoke({"query": query})).strip()
        except Exception as e:
            logger.error("Query reformulation failed for query %r: %s", query, e)
            return query  # fall back to original query

    def _get_prompt(self) -> ChatPromptTemplate:
        today = datetime.now().strftime("%Y-%m-%d")
        return ChatPromptTemplate.from_messages([
            (
                "system",
                f"Rewrite the user's query as a clear, specific search query. "
                f"Expand abbreviations and informal language. "
                f"Output one sentence only with no preamble. "
                f"Today's date is {today}."
            ),
            ("human", "User query: {query}\nRewritten search query:"),
        ])
