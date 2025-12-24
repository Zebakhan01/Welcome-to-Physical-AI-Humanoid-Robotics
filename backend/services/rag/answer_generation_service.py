"""
Answer Generation Service for Phase 2
Generates answers based on retrieved context with strict grounding enforcement
"""
import os
from typing import List, Dict, Any, Optional
import cohere
from pydantic import BaseModel
from ...utils.logger import logger


class AnswerRequest(BaseModel):
    """Request for answer generation"""
    question: str
    context: str
    max_tokens: int = 512
    temperature: float = 0.3
    model: str = "command-r-plus"


class AnswerResponse(BaseModel):
    """Response for answer generation"""
    answer: str
    question: str
    context_used: bool
    grounded: bool
    sources: List[str]


class AnswerGenerationService:
    """
    Service class for generating answers based on retrieved context
    """

    def __init__(self):
        """Initialize the answer generation service"""
        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            raise ValueError("COHERE_API_KEY environment variable is required")

        self.client = cohere.Client(api_key)
        self.default_model = os.getenv("COHERE_GENERATION_MODEL", "command-r-plus")
        logger.info("✅ Answer Generation Service initialized")

    async def generate_answer(self, request: AnswerRequest) -> AnswerResponse:
        """
        Generate an answer based on the question and context

        Args:
            request: Answer generation request with question and context

        Returns:
            Answer response with generated answer and metadata
        """
        try:
            # Check if context is provided
            if not request.context.strip():
                answer = "This information is not available in the textbook."
                return AnswerResponse(
                    answer=answer,
                    question=request.question,
                    context_used=False,
                    grounded=False,
                    sources=[]
                )

            # Create a prompt that enforces strict grounding in the context
            prompt = self._create_grounding_prompt(request.question, request.context)

            # Generate answer using Cohere
            response = self.client.generate(
                model=request.model,
                prompt=prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                stop_sequences=["\n\nQuestion:", "\n\nQ:", "\n\nA:"]
            )

            generated_text = response.generations[0].text.strip()

            # Validate that the answer is grounded in the context
            is_grounded = self._validate_answer_grounding(request.context, generated_text)

            # Extract sources if possible
            sources = self._extract_sources(request.context)

            logger.info(f"Generated answer for question: {request.question[:50]}...")
            return AnswerResponse(
                answer=generated_text,
                question=request.question,
                context_used=True,
                grounded=is_grounded,
                sources=sources
            )

        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            # Return fallback response if generation fails
            return AnswerResponse(
                answer="This information is not available in the textbook.",
                question=request.question,
                context_used=False,
                grounded=False,
                sources=[]
            )

    def _create_grounding_prompt(self, question: str, context: str) -> str:
        """
        Create a prompt that enforces strict grounding in the provided context

        Args:
            question: The question to answer
            context: The context to ground the answer in

        Returns:
            Formatted prompt string
        """
        prompt = f"""You are a helpful assistant that answers questions based strictly on the provided textbook content.

        TEXTBOOK CONTENT:
        {context}

        INSTRUCTIONS:
        1. Answer the question based ONLY on the textbook content provided above.
        2. If the answer is not available in the textbook content, respond with: "This information is not available in the textbook."
        3. Do not use any external knowledge or make assumptions beyond what's in the textbook.
        4. Provide accurate, concise answers based solely on the textbook information.

        QUESTION: {question}

        ANSWER:"""

        return prompt.strip()

    def _validate_answer_grounding(self, context: str, answer: str) -> bool:
        """
        Validate that the generated answer is grounded in the provided context

        Args:
            context: The context that was provided
            answer: The generated answer

        Returns:
            Boolean indicating if answer is grounded in context
        """
        # Simple validation: check if answer contains key terms from context
        # In a more sophisticated system, you might use semantic similarity
        context_lower = context.lower()
        answer_lower = answer.lower()

        # If the answer is the fallback message, it's not grounded
        if "not available in the textbook" in answer_lower:
            return False

        # Check if answer has some connection to context
        # This is a basic heuristic - in practice, you might want more sophisticated validation
        context_words = set(context_lower.split()[:50])  # Use first 50 words as representative
        answer_words = set(answer_lower.split())

        # If there's some overlap in terms, consider it grounded
        overlap = context_words.intersection(answer_words)
        return len(overlap) > 0 or len(context_words) == 0  # If context is empty, answer can't be grounded

    async def generate_answer_with_validation(self, question: str, context: str) -> AnswerResponse:
        """
        Generate an answer with additional validation for grounding

        Args:
            question: The question to answer
            context: The context to ground the answer in

        Returns:
            Answer response with validation results
        """
        # Generate the answer
        request = AnswerRequest(
            question=question,
            context=context
        )
        response = await self.generate_answer(request)

        # Additional validation could be added here
        # For example, checking if the answer directly references information in the context

        return response

    def _extract_sources(self, context: str) -> List[str]:
        """
        Extract potential sources from the context

        Args:
            context: The context text

        Returns:
            List of potential source identifiers
        """
        # In a more sophisticated implementation, this would extract actual source information
        # from the context metadata. For now, we'll return a simple placeholder.
        if context and len(context) > 0:
            # Return a simple identifier that context was used
            return ["textbook_content"]
        return []

    async def validate_answer_quality(self, question: str, answer: str, context: str) -> Dict[str, Any]:
        """
        Validate the quality and grounding of an answer

        Args:
            question: Original question
            answer: Generated answer
            context: Context used for generation

        Returns:
            Dictionary with validation results
        """
        try:
            # Check if answer contains the fallback message
            is_fallback = "not available in the textbook" in answer.lower()

            # Check if answer is related to the question
            question_relevance = self._check_question_relevance(question, answer)

            # Validate grounding
            is_grounded = self._validate_answer_grounding(context, answer)

            validation_results = {
                "is_fallback": is_fallback,
                "question_relevance": question_relevance,
                "is_grounded": is_grounded,
                "context_length": len(context),
                "answer_length": len(answer)
            }

            return validation_results

        except Exception as e:
            logger.error(f"Error validating answer quality: {str(e)}")
            return {
                "is_fallback": True,
                "question_relevance": False,
                "is_grounded": False,
                "error": str(e)
            }

    def _check_question_relevance(self, question: str, answer: str) -> bool:
        """
        Check if the answer is relevant to the question

        Args:
            question: Original question
            answer: Generated answer

        Returns:
            Boolean indicating if answer is relevant to question
        """
        # Simple heuristic: check if answer addresses key terms from question
        question_lower = question.lower()
        answer_lower = answer.lower()

        # Extract key terms from question (simple approach)
        question_terms = set(question_lower.split())

        # Check if answer contains any question terms or related concepts
        answer_words = set(answer_lower.split())

        # If there's significant overlap or answer is not the fallback, consider it relevant
        if "not available in the textbook" in answer_lower:
            return True  # Fallback is a valid response to the question

        overlap = question_terms.intersection(answer_words)
        return len(overlap) > 0 or len(question_terms) == 0  # If question is empty, any answer is relevant

    async def generate_answer_with_sources(self, question: str, context: str, sources: List[str]) -> AnswerResponse:
        """
        Generate an answer and include source information

        Args:
            question: The question to answer
            context: The context to ground the answer in
            sources: List of sources to reference

        Returns:
            Answer response with source information
        """
        # For now, this is similar to the basic generate_answer but could be extended
        # to include more sophisticated source tracking
        request = AnswerRequest(
            question=question,
            context=context
        )
        response = await self.generate_answer(request)

        # Update sources with provided sources
        response.sources = sources
        return response