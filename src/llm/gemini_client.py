"""
Client implementation for Google's Gemini models.
"""

import os
from typing import Dict, List, Optional, Union, Any, Generator
import logging
from .base import LLMClient
from pydantic import BaseModel
from google import genai
from google.genai import types
from config import models
import asyncio

logger = logging.getLogger(__name__)

class GeminiResponse(BaseModel):
    content: Optional[str] = None
    input_tokens: float
    output_tokens: float
    model: str
    cost: float
    success: bool = True
    error_message: Optional[str] = None


class GeminiStructuredResponse(BaseModel):
    structure: Optional[dict] = None
    input_tokens: float
    output_tokens: float
    model: str
    cost: float
    success: bool = True
    error_message: Optional[str] = None


class GeminiAsyncClient(LLMClient):
    """Async-capable client for Google's Gemini models."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.0-flash"):
        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "The API key must be provided either as an argument or via environment variable"
            )

        super().__init__()
        self.model = model
        self.client = genai.Client(api_key=api_key)

    async def generate(
        self,
        user_prompt: str,
        image,
        system_prompt: Optional[str] = None,
        max_tokens: int = 4048,
        temperature: float = 0.1,
    ) -> GeminiResponse:
        """Async generate text using Gemini."""
        try:
            # Prepare config
            if self.model in {"gemini-2.0-flash", "gemini-2.5-pro"}:
                model_config = types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                )
            elif self.model == "gemini-2.5-flash":
                model_config = types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                )

            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.models.generate_content(
                    model=self.model,
                    config=model_config,
                    contents=[image, user_prompt],
                ),
            )

            text = getattr(response, "text", None)
            if text is None:
                raise ValueError("Gemini response did not contain text output.")

            input_tok = response.usage_metadata.prompt_token_count
            output_tok = (
                response.usage_metadata.total_token_count - input_tok
            )
            cost = input_tok * (models[self.model]["input_cost"] / 1000000) + output_tok * (
                models[self.model]["output_cost"] / 1000000
            )

            return GeminiResponse(
                content=response.text,
                input_tokens=input_tok,
                output_tokens=output_tok,
                cost=cost,
                model=self.model,
                success=True,
            )

        except Exception as e:
            logger.error(f"Error generating text with Gemini: {str(e)}")
            return GeminiResponse(
                content=None,
                input_tokens=0,
                output_tokens=0,
                cost=0.0,
                model=self.model,
                success=False,
                error_message= f"Error generating text with Gemini from Google AI studio: {str(e)}",
            )

    async def generate_structured_response(
        self,
        user_prompt: str,
        structure,
        image,
        system_prompt: Optional[str] = None,
        max_tokens: int = 4048,
        temperature: float = 0.1,
    ) -> GeminiStructuredResponse:
        """Async structured response generation."""
        try:
            if self.model == "gemini-2.0-flash":
                model_config = types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    response_mime_type="application/json",
                    response_schema=structure,
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                )
            elif self.model in {"gemini-2.5-flash", "gemini-2.5-pro"}:
                model_config = types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    response_mime_type="application/json",
                    response_schema=structure,
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                )

            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.models.generate_content(
                    model=self.model,
                    config=model_config,
                    contents= [image, user_prompt],
                ),
            )

            input_tok = response.usage_metadata.prompt_token_count
            output_tok = (
                response.usage_metadata.total_token_count - input_tok
            )
            cost = input_tok * (models[self.model]["input_cost"] / 1000000) + output_tok * (
                models[self.model]["output_cost"] / 1000000
            )

            return GeminiStructuredResponse(
                structure=response.parsed,
                input_tokens=input_tok,
                output_tokens=output_tok,
                cost=cost,
                model=self.model,
                success=True,
            )

        except Exception as e:
            logger.error(f"Error generating structured data with Gemini: {str(e)}")
            return GeminiStructuredResponse(
                structure=None,
                input_tokens=0,
                output_tokens=0,
                cost=0.0,
                model=self.model,
                success=False,
                error_message=f"Error generating text with Gemini from Google AI studio: {str(e)}",
            )