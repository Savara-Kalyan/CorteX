import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from settings.config import settings

logger = logging.getLogger(__name__)


class QueryExpander:

    def __init__(self):
        cfg = settings.query_understanding
        self._llm = ChatOpenAI(model=cfg.model, temperature=cfg.temperature)

    async def expand(self, query: str) -> list[str]:
        try:
            chain = self._get_prompt() | self._llm | StrOutputParser()
            raw = (await chain.ainvoke({"query": query})).strip()
            return [q.strip() for q in raw.split("\n") if q.strip()][:3]
        except Exception as e:
            logger.error("Query expansion failed for query %r: %s", query, e)
            return []

    def _get_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            (
                "system",
                "Generate 3 alternative search queries for the same information. "
                "Use different words but preserve the original meaning. "
                "Output one query per line with no numbering or extra formatting."
            ),
            ("human", "Original: {query}"),
        ])
