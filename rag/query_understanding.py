import logging
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

from settings.config import settings

logger = logging.getLogger(__name__)

_FALLBACK = {"intent": "out_of_scope", "confidence": 0.0, "answerable": False}


class QueryIntentClassifier:

    def __init__(self):
        cfg = settings.query_understanding
        self._llm = ChatOpenAI(model=cfg.model, temperature=cfg.temperature)

    async def classify(self, query: str) -> dict:
        try:
            chain = self._get_prompt() | self._llm | JsonOutputParser()
            response = await chain.ainvoke({"query": query})
            if not isinstance(response, dict):
                return _FALLBACK
            return response
        except Exception as e:
            logger.error("Intent classification failed for query %r: %s", query, e)
            return _FALLBACK

    def _get_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a query intent classifier for a company knowledge base.\n"
                "Classify the query into exactly one of these domains:\n"
                "  - hr: queries about leave policies, payroll, onboarding, benefits, performance reviews, or HR processes\n"
                "  - engineering: queries about tech stack, system architecture, code practices, deployments, tooling, or engineering workflows\n"
                "  - culture: queries about company values, team norms, org structure, mission, events, or general company info\n"
                "  - out_of_scope: queries that do not belong to any of the above domains\n\n"
                "Respond with JSON only — no prose, no markdown:\n"
                '{{"intent": "<hr|engineering|culture|out_of_scope>", "confidence": <0.0-1.0>, "answerable": <true|false>}}'
            ),
            ("human", "Query: {query}"),
        ])


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
            return query

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


class QueryUnderstanding:

    def __init__(self):
        self.reformulator = QueryReformulator()
        self.expander = QueryExpander()
        self.classifier = QueryIntentClassifier()

    async def process(self, query: str) -> dict:
        try:
            reformulated = await self.reformulator.reformulate(query)
            expanded = await self.expander.expand(reformulated)
            intent_result = await self.classifier.classify(reformulated)

            return {
                "query": query,
                "reformulated": reformulated,
                "all_queries": [reformulated] + expanded[:2],
                "intent": intent_result.get("intent", "out_of_scope"),
                "answerable": intent_result.get("answerable", False),
                "confidence": intent_result.get("confidence", 0.0),
            }
        except Exception as e:
            logger.error("QueryUnderstanding.process failed for query %r: %s", query, e)
            raise
