"""
Google AI Gemini service for generating answers from retrieved context.

This service uses Google's Gemini 3.8 Flash model to generate grounded answers
based on document excerpts retrieved from the RAG pipeline.
"""

import os
import time
import asyncio
from typing import List, AsyncGenerator, Dict, Any, Tuple
import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai.types import GenerateContentConfig, ThinkingConfig
from app.services.vector_store import SearchResult


class GeminiGenerator:
    """Generate answers using Google AI Gemini 3.8 Flash with streaming."""

    def __init__(
        self,
        model_name: str = "gemini-3.8-flash",
        fallback_model_names: Tuple[str, ...] = ("gemini-3.6-flash", "gemini-3.5-flash"),
        temperature: float = 1.0,
        max_output_tokens: int = 8192,
    ):
        """
        Initialize the Gemini generator.

        Args:
            model_name: Gemini model to use (default: gemini-3.8-flash)
            fallback_model_names: Models tried in order when the previous one is
                overloaded, rate-limited, unreachable or no longer available
                (default: gemini-3.6-flash, then gemini-3.5-flash)
            temperature: Sampling temperature (default: 1.0)
            max_output_tokens: Maximum tokens in response (default: 8192)
        """
        # Load API key from environment
        api_key = os.getenv("GOOGLE_API_KEY")

        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY environment variable must be set. "
                "Get your API key from https://aistudio.google.com/apikey"
            )

        # Initialize Google AI client
        print(f"🔧 Initializing Google AI...")
        print(f"   Model: {model_name} (fallbacks: {', '.join(fallback_model_names) or 'none'})")

        # Create client with API key
        self.client = genai.Client(api_key=api_key)

        self.model_name = model_name
        self.fallback_model_names = tuple(fallback_model_names)
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

        print(f"✓ Google AI initialized successfully")

    @staticmethod
    def _is_overloaded(error: Exception) -> bool:
        """
        True for failures worth retrying on the next model: 503 overloaded,
        429 rate limited, 404 model not found (a model retired or not enabled
        for this key), or the connection dropping before a response
        ("Server disconnected without sending a response").
        """
        if isinstance(error, genai_errors.APIError):
            return error.code in (404, 429, 503)
        return isinstance(error, httpx.TransportError)

    def _models_to_try(self) -> List[str]:
        """The primary model, then each fallback in order, without repeats."""
        return list(dict.fromkeys([self.model_name, *self.fallback_model_names]))

    def _format_context(self, search_results: List[SearchResult]) -> str:
        """
        Format search results into a structured context string for the model.

        Args:
            search_results: List of search results from vector store

        Returns:
            Formatted context string with numbered excerpts
        """
        context_parts = []
        context_parts.append("=== Retrieved Document Excerpts ===\n")

        for i, result in enumerate(search_results, 1):
            # Parse title from filename (remove extension and double-underscore author suffix)
            title = result.filename.replace(".txt", "").split("__")[0].replace("_", " ")

            # Build source line with author and section if available
            source_line = f'\n[{i}] "{title}"'
            if result.author:
                source_line += f" by {result.author}"
            if result.section_label:
                source_line += f", {result.section_label}"

            context_parts.append(source_line)

            if result.reranking_score is not None:
                context_parts.append(
                    f"    Relevance Score: {result.reranking_score:.4f}"
                )

            # The excerpt is what the reader sees on the source card, so it is the
            # only citable text; the full passage is background for interpretation
            excerpt = result.excerpt_text or result.paragraph_text
            context_parts.append(f"\n    Excerpt (citable): {excerpt}")
            # Compared ignoring whitespace: the passage keeps the book's line
            # breaks while the excerpt's sentences are joined by spaces
            if result.paragraph_text and "".join(result.paragraph_text.split()) != "".join(
                excerpt.split()
            ):
                context_parts.append(
                    f"    Full passage (background only): {result.paragraph_text}"
                )

        return "\n".join(context_parts)

    def _create_system_prompt(self) -> str:
        """
        Create the system prompt for answer generation.

        Returns:
            System instruction string
        """
        return """You are a thoughtful philosophy guide. Answer questions using the provided source excerpts.

Guidelines:

1. **Explain clearly**: Translate philosophical ideas into plain, accessible language. If a quote uses archaic or dense phrasing, unpack what it means in modern terms.

2. **Be concise**: Aim for 2-3 paragraphs. Get to the point, then support it.

3. **Cite with numbered markers**: When referencing information from a source, include the citation number in brackets like [1], [2], etc. at the end of the relevant sentence or claim. Use separate brackets for multiple citations (e.g., [1] [3], not [1, 3]). Mention the author's name and/or work title naturally when first citing a source (e.g., "Emerson argues in Self Reliance... [1]"). Each source has a short "Excerpt" (what the reader sees beside the citation) and sometimes a "Full passage" for background. Only attach [n] to claims that source's Excerpt supports; use the full passage to understand context, not as the basis for a citation.

4. **Answer the question**: Lead with a clear response, then explain the reasoning from the sources.

5. **Stay grounded**: Only draw from the provided excerpts. If they don't fully address the question, say so briefly.

6. **Identify debates and tensions**: When philosophers directly engage with or critique each other's ideas, highlight this. If Nietzsche critiques Stoicism, or if two sources offer contrasting views, make the intellectual conversation explicit rather than presenting views in isolation.

7. **Own the answer**: Never start with meta-commentary like "The provided excerpts suggest...", "According to the sources...", "The consensus among these philosophers is...", or "These thinkers agree that...". Just answer the question directly, as if you're providing information you already know.

Tone: Warm but not gushing. Clear but not dry. Like explaining an interesting idea to a curious friend."""

    async def stream_answer(
        self,
        query: str,
        search_results: List[SearchResult],
        temperature: float = None,
        max_output_tokens: int = None,
    ) -> AsyncGenerator[str, None]:
        """
        Generate a streaming answer based on the query and retrieved context.

        Args:
            query: User's question
            search_results: List of relevant document excerpts
            temperature: Override default temperature
            max_output_tokens: Override default max tokens

        Yields:
            Text chunks from the model
        """
        # Format the context from search results
        context = self._format_context(search_results)

        # Create the system prompt and user prompt
        system_prompt = self._create_system_prompt()
        user_prompt = f"{context}\n\n=== Question ===\n{query}\n\n=== Answer ===\n"

        # Configure generation
        config = GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature if temperature is not None else self.temperature,
            max_output_tokens=(
                max_output_tokens
                if max_output_tokens is not None
                else self.max_output_tokens
            ),
            top_p=0.95,
            top_k=40,
            thinking_config=ThinkingConfig(
                thinking_level="medium"  # Medium thinking for accurate citations
            ),
        )

        # Generate streaming response. The SDK stream is synchronous, so it runs
        # in a background thread that hands chunks to this coroutine through an
        # asyncio queue. If the client disconnects, the coroutine is closed, the
        # finally block sets `cancelled`, and the thread stops reading from the
        # model instead of generating (and billing) the rest of the answer.
        try:
            import threading

            loop = asyncio.get_running_loop()
            chunk_queue: asyncio.Queue = asyncio.Queue()
            cancelled = threading.Event()
            error_holder = [None]

            def sync_stream():
                """Synchronous generator that yields chunks."""
                models = self._models_to_try()
                for i, model in enumerate(models):
                    started = False
                    stream = None
                    try:
                        stream = self.client.models.generate_content_stream(
                            model=model,
                            contents=user_prompt,
                            config=config,
                        )
                        for chunk in stream:
                            if cancelled.is_set():
                                return
                            if chunk.text:
                                started = True
                                yield chunk.text
                        return
                    except Exception as e:
                        # Only fall back if nothing was streamed yet, so the
                        # user never sees two half-answers spliced together
                        if started or i == len(models) - 1 or not self._is_overloaded(e):
                            raise
                        print(f"⚠️  {model} unavailable ({getattr(e, 'code', type(e).__name__)}), falling back to {models[i + 1]}")
                    finally:
                        close = getattr(stream, "close", None)
                        if callable(close):
                            close()

            def hand_off(item):
                try:
                    loop.call_soon_threadsafe(chunk_queue.put_nowait, item)
                except RuntimeError:
                    cancelled.set()  # event loop closed; nobody is listening

            def run_sync_stream():
                try:
                    for chunk in sync_stream():
                        if cancelled.is_set():
                            break
                        hand_off(chunk)
                except Exception as e:
                    error_holder[0] = e
                finally:
                    hand_off(None)  # Signal completion

            thread = threading.Thread(target=run_sync_stream, daemon=True)
            thread.start()

            try:
                while True:
                    chunk = await chunk_queue.get()
                    if chunk is None:
                        break
                    yield chunk
            finally:
                cancelled.set()

            if error_holder[0]:
                raise error_holder[0]

        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            print(f"❌ {error_msg}")
            import traceback

            traceback.print_exc()
            raise Exception(error_msg)

    def generate_answer(
        self,
        query: str,
        search_results: List[SearchResult],
        temperature: float = None,
        max_output_tokens: int = None,
    ) -> Dict[str, Any]:
        """
        Generate a complete (non-streaming) answer.

        Args:
            query: User's question
            search_results: List of relevant document excerpts
            temperature: Override default temperature
            max_output_tokens: Override default max tokens

        Returns:
            Dictionary with answer text and metadata
        """
        # Format the context from search results
        context = self._format_context(search_results)

        # Create the system prompt and user prompt
        system_prompt = self._create_system_prompt()
        user_prompt = f"{context}\n\n=== Question ===\n{query}\n\n=== Answer ===\n"

        # Configure generation
        config = GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature if temperature is not None else self.temperature,
            max_output_tokens=(
                max_output_tokens
                if max_output_tokens is not None
                else self.max_output_tokens
            ),
            top_p=0.95,
            top_k=40,
            thinking_config=ThinkingConfig(
                thinking_level="medium"  # Match the streaming path
            ),
        )

        # Generate response
        start_time = time.time()
        try:
            models = self._models_to_try()
            for i, model in enumerate(models):
                try:
                    response = self.client.models.generate_content(
                        model=model,
                        contents=user_prompt,
                        config=config,
                    )
                    break
                except Exception as e:
                    if i == len(models) - 1 or not self._is_overloaded(e):
                        raise
                    print(f"⚠️  {model} unavailable ({getattr(e, 'code', type(e).__name__)}), falling back to {models[i + 1]}")
            generation_time = time.time() - start_time

            return {
                "answer": response.text,
                "generation_time": generation_time,
                "model": model,
                "usage": {
                    "prompt_tokens": response.usage_metadata.prompt_token_count,
                    "completion_tokens": response.usage_metadata.candidates_token_count,
                    "total_tokens": response.usage_metadata.total_token_count,
                },
            }

        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            print(f"❌ {error_msg}")
            raise Exception(error_msg)
