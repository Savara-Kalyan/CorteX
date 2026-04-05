import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from settings.config import settings

logger = logging.getLogger(__name__)

FALLBACK = {"intent": "out_of_scope", "confidence": 0.0, "answerable": False}


class QueryIntentClassifier:

    def __init__(self):
        cfg = settings.query_understanding
        self._llm = ChatOpenAI(model=cfg.model, temperature=cfg.temperature)

    async def classify(self, query: str) -> dict:
        try:
            chain = self._get_prompt() | self._llm | JsonOutputParser()
            response = await chain.ainvoke({"query": query})
            if not isinstance(response, dict):
                logger.warning("Unexpected classifier response type: %s", type(response))
                return FALLBACK
            return response
        except Exception as e:
            logger.error("Intent classification failed for query %r: %s", query, e)
            return FALLBACK

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
                '{"intent": "<hr|engineering|culture|out_of_scope>", "confidence": <0.0-1.0>, "answerable": <true|false>}'
            ),
            ("human", "Query: {query}"),
        ])
