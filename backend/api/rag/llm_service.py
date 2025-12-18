from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from ...utils.logger import logger
from ...utils.config import config
import openai
import os
import asyncio

router = APIRouter()

# Set up OpenAI client
if config.OPENAI_API_KEY:
    openai.api_key = config.OPENAI_API_KEY
else:
    logger.warning("OPENAI_API_KEY not set. LLM functionality will not work.")

class GenerateAnswerRequest(BaseModel):
    query: str
    context: List[Dict[str, Any]]
    max_tokens: int = 500
    temperature: float = 0.7

class GenerateAnswerResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    query: str

async def generate_answer_with_context(query: str, context: List[Dict[str, Any]], max_tokens: int = 500, temperature: float = 0.7) -> GenerateAnswerResponse:
    """
    Generate an answer based on the query and provided context
    """
    try:
        if not config.OPENAI_API_KEY:
            raise HTTPException(status_code=500, detail="OpenAI API key not configured")

        # Format context for the LLM
        context_text = ""
        sources = []

        for item in context:
            content = item.get("content", "")
            title = item.get("title", "Unknown Source")
            score = item.get("score", 0)

            if content.strip():
                context_text += f"\n\nSource: {title}\nContent: {content}"
                sources.append({
                    "title": title,
                    "score": score,
                    "content_preview": content[:200] + "..." if len(content) > 200 else content
                })

        # Create the prompt for the LLM
        prompt = f"""Based on the following textbook content, please answer the question: '{query}'

        Context:
        {context_text}

        Please provide a comprehensive answer based on the provided context. If the context doesn't contain sufficient information to answer the question, please state that clearly."""

        # Use OpenAI's async client
        client = openai.AsyncOpenAI(api_key=config.OPENAI_API_KEY)

        response = await client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert assistant for the Physical AI & Humanoid Robotics textbook. Provide accurate, helpful answers based on the provided context."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )

        answer = response.choices[0].message.content.strip()

        return GenerateAnswerResponse(
            answer=answer,
            sources=sources,
            query=query
        )

    except Exception as e:
        logger.error(f"Error generating answer with LLM: {str(e)}")
        raise HTTPException(status_code=500, detail=f"LLM answer generation failed: {str(e)}")

@router.post("/generate-answer", response_model=GenerateAnswerResponse)
async def generate_answer_endpoint(request: GenerateAnswerRequest):
    """
    Generate an answer based on the query and provided context
    """
    try:
        logger.info(f"LLM answer generation requested for query: {request.query}")

        result = await generate_answer_with_context(
            query=request.query,
            context=request.context,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )

        return result

    except Exception as e:
        logger.error(f"Error in LLM answer generation endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"LLM answer generation failed: {str(e)}")

# Fallback implementation using a simple template if OpenAI is not available
def generate_simple_answer_with_context(query: str, context: List[Dict[str, Any]]) -> GenerateAnswerResponse:
    """
    Generate a simple answer based on context when OpenAI is not available
    """
    if not context:
        return GenerateAnswerResponse(
            answer=f"I couldn't find specific information about '{query}' in the textbook. Please try rephrasing your question.",
            sources=[],
            query=query
        )

    # Format the retrieved content as context
    context_parts = []
    sources = []

    for result in context:
        content = result.get("content", "")
        title = result.get("title", "Unknown")
        score = result.get("score", 0)

        if content.strip():  # Only add non-empty content
            context_parts.append(f"Source: {title}\nContent: {content}")
            sources.append({
                "title": title,
                "score": score,
                "content_preview": content[:200] + "..." if len(content) > 200 else content
            })

    context_text = "\n\n".join(context_parts)

    # Generate a response based on the context and user query
    answer = f"Based on the textbook content, here's information related to your query '{query}':\n\n{context_text}"

    return GenerateAnswerResponse(
        answer=answer,
        sources=sources,
        query=query
    )