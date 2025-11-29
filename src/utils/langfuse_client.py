"""
Langfuse integration for comprehensive workflow tracing and observability.
"""

import os
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
from langfuse import Langfuse, observe
from src.config import settings


class LangfuseClient:
    """
    Wrapper for Langfuse client with enhanced tracing capabilities.
    Provides comprehensive observability for the multi-agent system.
    """

    def __init__(self):
        """Initialize Langfuse client with configuration."""
        self.client = None
        self.enabled = self._initialize_client()

    def _initialize_client(self) -> bool:
        """Initialize the Langfuse client with proper configuration."""
        try:
            # Check if credentials are available
            secret_key = settings.langfuse_secret_key or os.getenv('LANGFUSE_SECRET_KEY')
            public_key = settings.langfuse_public_key or os.getenv('LANGFUSE_PUBLIC_KEY')
            host = settings.langfuse_base_url or os.getenv('LANGFUSE_BASE_URL')

            if not all([secret_key, public_key, host]):
                print("Langfuse credentials not found. Observability disabled.")
                return False

            # Initialize client with correct parameters
            self.client = Langfuse(
                secret_key=secret_key,
                public_key=public_key,
                host=host  # Note: parameter is 'host', not 'base_url'
            )

            print("Langfuse client initialized successfully")
            return True

        except Exception as e:
            print(f"Failed to initialize Langfuse client: {e}")
            print("Continuing without observability...")
            return False

    @observe
    def create_trace(
        self,
        name: str,
        input: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Create a new trace for tracking a complete workflow."""
        if not self.enabled or not self.client:
            print("📝 [Langfuse] Creating trace (dummy mode)")
            return self._create_dummy_trace()

        try:
            print(f"📝 [Langfuse] Creating trace: {name}")

            # Get the actual trace created by @observe decorator
            try:
                # Try to get the current trace from the Langfuse client
                current_trace_id = getattr(self.client, '_current_trace_id', None)
                if current_trace_id:
                    trace_context = EnhancedTraceContext(
                        name=name,
                        input=input,
                        user_id=user_id,
                        metadata=metadata or {},
                        client=self.client,
                        trace_id=current_trace_id
                    )
                else:
                    # Fallback to basic trace context
                    trace_context = EnhancedTraceContext(
                        name=name,
                        input=input,
                        user_id=user_id,
                        metadata=metadata or {},
                        client=self.client
                    )
            except Exception:
                # Fallback to basic trace context
                trace_context = EnhancedTraceContext(
                    name=name,
                    input=input,
                    user_id=user_id,
                    metadata=metadata or {},
                    client=self.client
                )

            print(f"✅ [Langfuse] Enhanced trace context created")
            return trace_context

        except Exception as e:
            print(f"❌ [Langfuse] Failed to create trace: {e}")
            return self._create_dummy_trace()

    def _create_dummy_trace(self):
        """Create a dummy trace object for backward compatibility."""
        class DummyTrace:
            def span(self, **kwargs):
                return self._create_dummy_observation()
            def generation(self, **kwargs):
                return self._create_dummy_observation()
            def event(self, **kwargs):
                return self._create_dummy_observation()
            def update(self, **kwargs):
                pass
            def __getattr__(self, name):
                return lambda *args, **kwargs: self._create_dummy_observation()

            def _create_dummy_observation(self):
                class DummyObservation:
                    def update(self, **kwargs):
                        pass
                    def end(self):
                        pass
                    def __enter__(self):
                        return self
                    def __exit__(self, *args):
                        pass
                return DummyObservation()

        return DummyTrace()

    def add_observation(
        self,
        trace,
        name: str,
        observation_type: str = "span",
        input: Optional[str] = None,
        output: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ):
        """Add an observation to an existing trace."""
        if not self.enabled or not self.client or not trace:
            return None

        try:
            observation_data = {
                "name": name,
                "input": input,
                "output": output,
                "metadata": metadata or {}
            }

            if observation_type == "span":
                observation = trace.span(**observation_data)
            elif observation_type == "generation":
                observation = trace.generation(**observation_data)
            elif observation_type == "event":
                observation = trace.event(**observation_data)
            else:
                # Fallback to span if observation type not found
                observation = trace.span(**observation_data)

            print(f"✅ [Langfuse] {observation_type.capitalize()} created: {name}")
            return observation
        except Exception as e:
            print(f"❌ [Langfuse] Failed to add {observation_type}: {e}")
            return None

    def log_agent_execution(
        self,
        trace,
        agent_name: str,
        agent_type: str,
        input_data: str,
        output_data: str,
        execution_time: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log agent execution with performance metrics."""
        span_metadata = {
            "agent_name": agent_name,
            "agent_type": agent_type,
            "execution_time_seconds": execution_time,
            "input_length": len(input_data) if input_data else 0,
            "output_length": len(output_data) if output_data else 0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "performance_tier": "fast" if execution_time < 1.0 else "medium" if execution_time < 3.0 else "slow"
        }

        if metadata:
            span_metadata.update(metadata)

        return self.add_observation(
            trace=trace,
            name=f"{agent_name}_execution",
            observation_type="span",
            input=input_data[:1000] + "..." if input_data and len(input_data) > 1000 else input_data,
            output=output_data[:1000] + "..." if output_data and len(output_data) > 1000 else output_data,
            metadata=span_metadata
        )

    def log_classification_result(
        self,
        trace,
        query: str,
        predicted_class: str,
        confidence: float,
        all_scores: Dict[str, float]
    ):
        """Log intent classification results with confidence scores."""
        # Calculate additional metrics
        max_score = max(all_scores.values()) if all_scores else 0.0
        score_gap = max_score - (sorted(all_scores.values())[-2] if len(all_scores) > 1 else 0.0)
        is_high_confidence = confidence > settings.confidence_threshold

        return self.add_observation(
            trace=trace,
            name="intent_classification",
            observation_type="event",
            input=query[:500] + "..." if query and len(query) > 500 else query,
            output=predicted_class,
            metadata={
                "confidence": confidence,
                "max_score": max_score,
                "score_gap": score_gap,
                "all_scores": all_scores,
                "threshold": settings.confidence_threshold,
                "classification_method": "semantic_similarity",
                "is_high_confidence": is_high_confidence,
                "alternative_departments": sorted(all_scores.keys(), key=lambda x: all_scores[x], reverse=True)[1:3] if len(all_scores) > 1 else []
            }
        )

    def log_rag_retrieval(
        self,
        trace,
        query: str,
        retrieved_docs: List[str],
        similarity_scores: List[float],
        retrieval_time: float
    ):
        """Log RAG document retrieval results."""
        # Calculate detailed metrics
        avg_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.0
        max_similarity = max(similarity_scores) if similarity_scores else 0.0
        min_similarity = min(similarity_scores) if similarity_scores else 0.0

        # Categorize retrieval quality
        if avg_similarity > 0.8:
            quality_tier = "excellent"
        elif avg_similarity > 0.6:
            quality_tier = "good"
        elif avg_similarity > 0.4:
            quality_tier = "fair"
        else:
            quality_tier = "poor"

        # Create document summaries for metadata
        doc_summaries = []
        for i, doc in enumerate(retrieved_docs[:3]):  # Only include first 3 docs
            similarity_score = similarity_scores[i] if i < len(similarity_scores) else 0.0
            doc_preview = doc[:100] + "..." if doc and len(doc) > 100 else doc
            doc_summaries.append({
                "doc_index": i,
                "similarity_score": similarity_score,
                "preview": doc_preview
            })

        return self.add_observation(
            trace=trace,
            name="rag_retrieval",
            observation_type="span",
            input=query[:500] + "..." if query and len(query) > 500 else query,
            output=f"Retrieved {len(retrieved_docs)} documents with avg similarity {avg_similarity:.3f}",
            metadata={
                "num_documents": len(retrieved_docs),
                "similarity_scores": similarity_scores,
                "average_similarity": avg_similarity,
                "max_similarity": max_similarity,
                "min_similarity": min_similarity,
                "retrieval_time_seconds": retrieval_time,
                "embedding_model": settings.embedding_model,
                "quality_tier": quality_tier,
                "document_previews": doc_summaries,
                "similarity_distribution": {
                    "high_confidence_docs": sum(1 for s in similarity_scores if s > 0.8),
                    "medium_confidence_docs": sum(1 for s in similarity_scores if 0.5 <= s <= 0.8),
                    "low_confidence_docs": sum(1 for s in similarity_scores if s < 0.5)
                }
            }
        )

    def log_quality_evaluation(
        self,
        trace,
        query: str,
        answer: str,
        context: str,
        quality_scores: Dict[str, float],
        overall_score: float
    ):
        """Log response quality evaluation results."""
        # Categorize overall quality
        if overall_score >= 8.5:
            quality_tier = "excellent"
        elif overall_score >= 7.0:
            quality_tier = "good"
        elif overall_score >= 5.0:
            quality_tier = "acceptable"
        else:
            quality_tier = "needs_improvement"

        return self.add_observation(
            trace=trace,
            name="quality_evaluation",
            observation_type="generation",
            input=query[:500] + "..." if query and len(query) > 500 else query,
            output=answer[:1000] + "..." if answer and len(answer) > 1000 else answer,
            metadata={
                "context_length": len(context),
                "context_preview": context[:200] + "..." if context and len(context) > 200 else context,
                "answer_length": len(answer),
                "quality_scores": quality_scores,
                "overall_score": overall_score,
                "quality_tier": quality_tier,
                "evaluation_dimensions": settings.quality_dimensions,
                "evaluator_model": settings.evaluator_model,
                "weakest_dimension": min(quality_scores.keys(), key=lambda x: quality_scores[x]) if quality_scores else None,
                "strongest_dimension": max(quality_scores.keys(), key=lambda x: quality_scores[x]) if quality_scores else None,
                "score_variance": max(quality_scores.values()) - min(quality_scores.values()) if quality_scores else 0.0
            }
        )

    def log_error(
        self,
        trace,
        error_message: str,
        error_type: str,
        context: Optional[Dict[str, Any]] = None
    ):
        """Log error events for debugging and monitoring."""
        return self.add_observation(
            trace=trace,
            name="error",
            observation_type="event",
            input=str(context)[:500] if context else "",
            output=error_message,
            metadata={
                "error_type": error_type,
                "error_severity": self._categorize_error_severity(error_type),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error_category": self._categorize_error_type(error_type),
                **(context or {})
            }
        )

    def _categorize_error_severity(self, error_type: str) -> str:
        """Categorize error severity based on error type."""
        critical_errors = ["timeout", "connection_error", "api_key_error", "authentication_error"]
        warning_errors = ["low_confidence", "fallback_used", "retrieval_failed"]

        if any(critical in error_type.lower() for critical in critical_errors):
            return "critical"
        elif any(warning in error_type.lower() for warning in warning_errors):
            return "warning"
        else:
            return "info"

    def _categorize_error_type(self, error_type: str) -> str:
        """Categorize error type for better debugging."""
        if "classification" in error_type.lower():
            return "intent_classification"
        elif "retrieval" in error_type.lower():
            return "document_retrieval"
        elif "llm" in error_type.lower() or "generation" in error_type.lower():
            return "llm_generation"
        elif "agent" in error_type.lower():
            return "agent_processing"
        elif "orchestrator" in error_type.lower():
            return "orchestration"
        else:
            return "general"

    def log_llm_call(
        self,
        trace,
        model_name: str,
        prompt: str,
        response: str,
        token_usage: Dict[str, int],
        response_time: float,
        temperature: float = None,
        max_tokens: int = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log LLM API calls with detailed metrics."""
        # Calculate cost estimation (rough estimate)
        estimated_cost = self._estimate_llm_cost(model_name, token_usage)

        llm_metadata = {
            "model_name": model_name,
            "prompt_length": len(prompt),
            "response_length": len(response),
            "total_tokens": token_usage.get("total_tokens", 0),
            "prompt_tokens": token_usage.get("prompt_tokens", 0),
            "completion_tokens": token_usage.get("completion_tokens", 0),
            "response_time_seconds": response_time,
            "tokens_per_second": (token_usage.get("total_tokens", 0) / response_time) if response_time > 0 else 0,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "estimated_cost_usd": estimated_cost,
            "performance_tier": "fast" if response_time < 2.0 else "medium" if response_time < 5.0 else "slow",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        if metadata:
            llm_metadata.update(metadata)

        return self.add_observation(
            trace=trace,
            name=f"llm_call_{model_name.replace('/', '_').replace('-', '_')}",
            observation_type="generation",
            input=prompt[:2000] + "..." if prompt and len(prompt) > 2000 else prompt,
            output=response[:2000] + "..." if response and len(response) > 2000 else response,
            metadata=llm_metadata
        )

    def _estimate_llm_cost(self, model_name: str, token_usage: Dict[str, int]) -> float:
        """Estimate cost for LLM call in USD (rough estimates)."""
        # Rough cost per 1M tokens (update with actual pricing)
        cost_per_million = {
            "gpt-3.5-turbo": 0.50,
            "gpt-4": 30.0,
            "gpt-4-turbo": 10.0,
            "anthropic-claude-3-sonnet": 15.0,
            "anthropic-claude-3-haiku": 1.0,
        }

        # Extract model family
        model_family = None
        for family in cost_per_million.keys():
            if family in model_name.lower():
                model_family = family
                break

        if not model_family:
            return 0.0  # Unknown model, can't estimate cost

        cost_per_token = cost_per_million[model_family] / 1_000_000
        total_tokens = token_usage.get("total_tokens", 0)

        return total_tokens * cost_per_token

    def flush(self) -> None:
        """Flush any pending traces to Langfuse."""
        if self.enabled and self.client:
            try:
                if hasattr(self.client, 'flush'):
                    self.client.flush()
                else:
                    print("Langfuse client does not have flush method available")
            except Exception as e:
                print(f"Failed to flush Langfuse: {e}")


class EnhancedTraceContext:
    """Enhanced trace context that provides real Langfuse functionality."""

    def __init__(self, name: str, input: str, user_id: str, metadata: Dict[str, Any], client, trace_id: str = None):
        self.name = name
        self.input = input
        self.user_id = user_id
        self.metadata = metadata
        self.client = client
        self.trace_id = trace_id
        self.observations = []

    def span(self, **kwargs):
        """Create a real span observation."""
        try:
            # Create span via API if possible
            if hasattr(self.client, '_api') and self.trace_id:
                span_data = {
                    "trace_id": self.trace_id,
                    "name": kwargs.get('name', 'unnamed'),
                    "input": kwargs.get('input'),
                    "output": kwargs.get('output'),
                    "metadata": kwargs.get('metadata', {}),
                    "level": "DEFAULT"
                }
                span = self.client.span(**span_data)
                self.observations.append(span)
                print(f"✅ [Langfuse] Real span created: {kwargs.get('name', 'unnamed')}")
                return RealSpan(span, self.client)
            else:
                # Fallback to enhanced context manager
                span_context = EnhancedSpanContext(kwargs, self.client, self.trace_id)
                self.observations.append(span_context)
                print(f"✅ [Langfuse] Enhanced span context created: {kwargs.get('name', 'unnamed')}")
                return span_context
        except Exception as e:
            print(f"❌ [Langfuse] Failed to create span: {e}")
            return EnhancedSpanContext({}, self.client, self.trace_id)

    def generation(self, **kwargs):
        """Create a real generation observation."""
        try:
            # Create generation via API if possible
            if hasattr(self.client, '_api') and self.trace_id:
                generation_data = {
                    "trace_id": self.trace_id,
                    "name": kwargs.get('name', 'unnamed'),
                    "input": kwargs.get('input'),
                    "output": kwargs.get('output'),
                    "metadata": kwargs.get('metadata', {}),
                    "level": "DEFAULT"
                }
                generation = self.client.generation(**generation_data)
                self.observations.append(generation)
                print(f"✅ [Langfuse] Real generation created: {kwargs.get('name', 'unnamed')}")
                return RealGeneration(generation, self.client)
            else:
                # Fallback to enhanced context manager
                generation_context = EnhancedGenerationContext(kwargs, self.client, self.trace_id)
                self.observations.append(generation_context)
                print(f"✅ [Langfuse] Enhanced generation context created: {kwargs.get('name', 'unnamed')}")
                return generation_context
        except Exception as e:
            print(f"❌ [Langfuse] Failed to create generation: {e}")
            return EnhancedGenerationContext({}, self.client, self.trace_id)

    def event(self, **kwargs):
        """Create a real event observation."""
        try:
            # Create event via API if possible
            if hasattr(self.client, '_api') and self.trace_id:
                event_data = {
                    "trace_id": self.trace_id,
                    "name": kwargs.get('name', 'unnamed'),
                    "input": kwargs.get('input'),
                    "output": kwargs.get('output'),
                    "metadata": kwargs.get('metadata', {}),
                    "level": "DEFAULT"
                }
                event = self.client.event(**event_data)
                self.observations.append(event)
                print(f"✅ [Langfuse] Real event created: {kwargs.get('name', 'unnamed')}")
                return RealEvent(event, self.client)
            else:
                # Fallback to enhanced context manager
                event_context = EnhancedEventContext(kwargs, self.client, self.trace_id)
                self.observations.append(event_context)
                print(f"✅ [Langfuse] Enhanced event context created: {kwargs.get('name', 'unnamed')}")
                return event_context
        except Exception as e:
            print(f"❌ [Langfuse] Failed to create event: {e}")
            return EnhancedEventContext({}, self.client, self.trace_id)

    def update(self, **kwargs):
        """Update the trace with final data."""
        self.metadata.update(kwargs)
        print(f"🔄 [Langfuse] Trace updated with: {list(kwargs.keys())}")

        # Try to update the actual trace via API
        if hasattr(self.client, '_api') and self.trace_id:
            try:
                update_data = {
                    "id": self.trace_id,
                    "output": kwargs.get('output'),
                    "metadata": kwargs.get('metadata', {}),
                    "level": "DEFAULT"
                }
                self.client.trace(**update_data)
                print(f"✅ [Langfuse] Real trace updated successfully")
            except Exception as api_error:
                print(f"⚠️ [Langfuse] API trace update failed: {api_error}")


class RealSpan:
    """Wrapper for real Langfuse span objects."""
    def __init__(self, span, client):
        self.span = span
        self.client = client

    def update(self, **kwargs):
        try:
            if hasattr(self.span, 'update'):
                self.span.update(**kwargs)
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class RealGeneration:
    """Wrapper for real Langfuse generation objects."""
    def __init__(self, generation, client):
        self.generation = generation
        self.client = client

    def update(self, **kwargs):
        try:
            if hasattr(self.generation, 'update'):
                self.generation.update(**kwargs)
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class RealEvent:
    """Wrapper for real Langfuse event objects."""
    def __init__(self, event, client):
        self.event = event
        self.client = client

    def update(self, **kwargs):
        try:
            if hasattr(self.event, 'update'):
                self.event.update(**kwargs)
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class EnhancedSpanContext:
    """Enhanced context manager for span operations."""
    def __init__(self, span_data, client, trace_id):
        self.span_data = span_data
        self.client = client
        self.trace_id = trace_id

    def update(self, **kwargs):
        self.span_data.update(kwargs)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class EnhancedGenerationContext:
    """Enhanced context manager for generation operations."""
    def __init__(self, generation_data, client, trace_id):
        self.generation_data = generation_data
        self.client = client
        self.trace_id = trace_id

    def update(self, **kwargs):
        self.generation_data.update(kwargs)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class EnhancedEventContext:
    """Enhanced context manager for event operations."""
    def __init__(self, event_data, client, trace_id):
        self.event_data = event_data
        self.client = client
        self.trace_id = trace_id

    def update(self, **kwargs):
        self.event_data.update(kwargs)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


# Maintain backward compatibility
TraceContext = EnhancedTraceContext


# Global Langfuse client instance
langfuse_client = LangfuseClient()