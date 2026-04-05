import logging
from .query_expansion import QueryExpander
from .query_reformulation import QueryReformulator
from .query_classification import QueryIntentClassifier

logger = logging.getLogger(__name__)


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
