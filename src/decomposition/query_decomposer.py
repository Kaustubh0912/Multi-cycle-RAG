import re
from typing import Dict, List, Optional

from ..core.interfaces import LLMInterface, QueryDecomposerInterface
from ..llm.github_llm import GitHubLLM


class SmartQueryDecomposer(QueryDecomposerInterface):
    """Smart Query Decomposition with context awareness"""

    def __init__(self, llm: Optional[LLMInterface] = None):
        self.llm = llm or GitHubLLM()

        # Keywords that suggest complex queries needing decomposition
        self.complexity_indicators = [
            "and",
            "or",
            "both",
            "either",
            "compare",
            "contrast",
            "versus",
            "vs",
            "difference",
            "similarities",
            "benefits and risks",
            "pros and cons",
            "advantages and disadvantages",
            "before and after",
            "cause and effect",
            "how many",
            "what are",
            "list",
            "multiple",
            "several",
            "various",
        ]

    async def should_decompose(self, query: str) -> bool:
        """Determine if a query needs decomposition using smart heuristics"""
        query_lower = query.lower()

        # Check for complexity indicators
        complexity_score = sum(
            1
            for indicator in self.complexity_indicators
            if indicator in query_lower
        )

        # Check for multiple question marks or conjunctions
        question_marks = query.count("?")
        conjunctions = len(
            re.findall(r"\b(and|or|but|while|whereas)\b", query_lower)
        )

        # Check query length (longer queries often benifit from decomposition)
        word_count = len(query.split())

        # Scoring system
        should_decompose = (
            complexity_score >= 2  # Multiple complexity indicators
            or question_marks > 1  # Multiple questions
            or conjunctions >= 2  # Multiple conjunctions
            or word_count > 15  # Long query
        )

        return should_decompose

    async def decompose_query(
        self, query: str, context: Optional[str] = None
    ) -> List[str]:
        """Decompose a complex query into more focused sub-queries"""

        # First check if decomposition is needed
        if not await self.should_decompose(query):
            return [query]  # Return original query if no decomposition needed

        decomposition_prompt = self._create_decomposition_prompt(query, context)

        try:
            response = await self.llm.generate(
                decomposition_prompt, temperature=0.3
            )
            sub_queries = self._parse_sub_queries(response)

            # Ensure we have valid sub-queries
            if not sub_queries or len(sub_queries) == 1:
                return [query]
            return sub_queries

        except Exception as e:
            print(f"Decomposition failed: {e}, using original query")
            return [query]

    def _create_decomposition_prompt(
        self, query: str, context: Optional[str] = None
    ) -> str:
        """Create a smart decomposition prompt"""

        context_section = ""
        if context:
            context_section = f"""
    Context from previous conversation:
    {context}
    """

        return f"""You are an expert at breaking down complex questions into simpler, focused sub-questions.

            {context_section}

            Original Question: {query}

            Instructions:
            - Break this question into 2-5 focused sub-questions that can be answered independently
            - Each sub-question should be specific and searchable
            - Ensure sub-questions cover all aspects of the original question
            - If the question references previous context, incorporate that context into the sub-questions
            - Make each sub-question standalone (no pronouns like "it", "they", "this")
            - Number each sub-question clearly

            Format your response as:
            1. [First sub-question]
            2. [Second sub-question]
            3. [Third sub-question]
            ...

            Sub-questions:"""

    def _parse_sub_queries(self, response: str) -> List[str]:
        """Parse sub-queries from LLM response"""
        lines = response.strip().split("\n")
        sub_queries = []

        for line in lines:
            line = line.strip()

            # Match numbered lines (1. 2. etc.)
            match = re.match(r"^\d+\.\s(.+)", line)
            if match:
                sub_query = match.group(1).strip()
                if sub_query and len(sub_query) > 5:  # Minimum length check
                    sub_queries.append(sub_query)

        return sub_queries


class ContextAwareDecomposer(SmartQueryDecomposer):
    """Context-aware decomposer for multi-turn conversations"""

    def __init__(self, llm: Optional[LLMInterface] = None):
        super().__init__(llm)
        self.conversation_history: List[Dict[str, str]] = []

    def add_to_history(self, query: str, response: str):
        """Add interaction to conversation history"""
        self.conversation_history.append({"query": query, "response": response})

        # Keep only last 3 interactions to avoid context bloat
        if len(self.conversation_history) > 3:
            self.conversation_history = self.conversation_history[-3:]

    async def decompose_query(
        self, query: str, context: Optional[str] = None
    ) -> List[str]:
        """Context-aware decomposition using conversation history"""

        # Build context from conversation history
        if self.conversation_history and not context:
            context_parts = []
            for interaction in self.conversation_history[
                -2:
            ]:  # Last 2 interactions
                context_parts.append(f"Q: {interaction['query']}")
                context_parts.append(
                    f"A: {interaction['response'][:200]}..."
                )  # Truncate long responses

            context = "\n".join(context_parts)

        return await super().decompose_query(query, context)
