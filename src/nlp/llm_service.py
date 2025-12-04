"""LLM service for generating responses with multiple provider support."""
import os
from typing import Dict, Any, List, Optional, Iterator
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        stream: bool = False
    ) -> str:
        """Generate response."""
        pass
    
    @abstractmethod
    def generate_stream(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> Iterator[str]:
        """Generate streaming response."""
        pass

class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider."""
    
    def __init__(self, api_key: str, model: str = 'gpt-4o-mini'):
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
            self.model = model
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        stream: bool = False
    ) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream
            )
            if stream:
                return response  # For streaming, return the stream object
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    def generate_stream(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> Iterator[str]:
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"OpenAI streaming error: {e}")
            raise

class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider."""
    
    def __init__(self, api_key: str, model: str = 'claude-3-haiku-20240307'):
        try:
            from anthropic import Anthropic
            self.client = Anthropic(api_key=api_key)
            self.model = model
        except ImportError:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        stream: bool = False
    ) -> str:
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise
    
    def generate_stream(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> Iterator[str]:
        try:
            with self.client.messages.stream(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            ) as stream:
                for chunk in stream:
                    if chunk.type == 'content_block_delta':
                        yield chunk.delta.text
        except Exception as e:
            logger.error(f"Anthropic streaming error: {e}")
            raise

class GroqProvider(LLMProvider):
    """Groq provider for fast inference."""
    
    def __init__(self, api_key: str, model: str = 'llama-3.1-70b-versatile'):
        try:
            from groq import Groq
            self.client = Groq(api_key=api_key)
            self.model = model
        except ImportError:
            raise ImportError("groq package not installed. Run: pip install groq")
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        stream: bool = False
    ) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            raise
    
    def generate_stream(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> Iterator[str]:
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"Groq streaming error: {e}")
            raise

class LLMService:
    """Service for managing LLM interactions with multiple providers."""
    
    def __init__(
        self,
        provider: str = 'openai',
        model: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        self.provider_name = provider
        self.model = model
        self.temperature = 0.7
        self.max_tokens = 1000
        self.api_key = api_key or self._get_api_key(provider)
        self.provider = self._init_provider(provider, self.api_key, model)
    
    def _get_api_key(self, provider: str) -> Optional[str]:
        key_map = {
            'openai': 'OPENAI_API_KEY',
            'anthropic': 'ANTHROPIC_API_KEY',
            'groq': 'GROQ_API_KEY'
        }
        return os.getenv(key_map.get(provider, ''))
    
    def _init_provider(
        self,
        provider: str,
        api_key: str,
        model: Optional[str]
    ) -> LLMProvider:
        provider = provider.lower()
        
        if provider == 'openai':
            model = model or os.getenv('LLM_MODEL', 'gpt-4o-mini')
            return OpenAIProvider(api_key, model)
        elif provider == 'anthropic':
            model = model or 'claude-3-haiku-20240307'
            return AnthropicProvider(api_key, model)
        elif provider == 'groq':
            model = model or 'llama-3.1-70b-versatile'
            return GroqProvider(api_key, model)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> str:
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        try:
            return self.provider.generate(prompt, temp, tokens, stream)
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise
    
    def generate_stream(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Iterator[str]:
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        try:
            yield from self.provider.generate_stream(prompt, temp, tokens)
        except Exception as e:
            logger.error(f"LLM streaming failed: {e}")
            raise
    
    def create_rag_prompt(
        self,
        question: str,
        context: List[str],
        system_prompt: Optional[str] = None
    ) -> str:
        if not system_prompt:
            system_prompt = (
                "You are a helpful AI assistant. Answer the question based on the "
                "provided context. If the context doesn't contain enough information "
                "to answer the question, say so clearly."
            )
        
        context_text = "\n\n".join([f"[{i+1}] {chunk}" for i, chunk in enumerate(context)])
        
        prompt = f"""{system_prompt}

Context:
{context_text}

Question: {question}

Answer: """
        
        return prompt