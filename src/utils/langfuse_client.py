"""
Langfuse integration for comprehensive workflow tracing and observability.
"""

import os
from typing import Optional, Dict, Any, List
from datetime import datetime
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

    @observe(name="multi_agent_query_processing")
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
            # Use the @observe decorator for automatic trace creation
            # This will handle the trace creation automatically
            context = {
                'name': name,
                'input': input,
                'user_id': user_id,
                'metadata': metadata or {}
            }

            # Create a simple trace context object
            class TraceContext:
                def __init__(self, client, context):
                    self.client = client
                    self.context = context
                    self.observations = []
                    print(f"✅ [Langfuse] Trace created successfully")

                def span(self, **kwargs):
                    """Create a span observation."""
                    print(f"📊 [Langfuse] Creating span: {kwargs.get('name', 'unnamed')}")
                    observation = {
                        'type': 'span',
                        'data': kwargs
                    }
                    self.observations.append(observation)
                    return self._create_observation_wrapper(observation)

                def generation(self, **kwargs):
                    """Create a generation observation."""
                    print(f"🤖 [Langfuse] Creating generation: {kwargs.get('name', 'unnamed')}")
                    observation = {
                        'type': 'generation',
                        'data': kwargs
                    }
                    self.observations.append(observation)
                    return self._create_observation_wrapper(observation)

                def event(self, **kwargs):
                    """Create an event observation."""
                    print(f"📅 [Langfuse] Creating event: {kwargs.get('name', 'unnamed')}")
                    observation = {
                        'type': 'event',
                        'data': kwargs
                    }
                    self.observations.append(observation)
                    return self._create_observation_wrapper(observation)

                def update(self, **kwargs):
                    """Update the trace with final data."""
                    print(f"🔄 [Langfuse] Updating trace with: {list(kwargs.keys())}")
                    self.context.update(kwargs)

                def _create_observation_wrapper(self, observation):
                    """Create a wrapper for individual observations."""
                    class ObservationWrapper:
                        def __init__(self, obs):
                            self.observation = obs
                        def update(self, **kwargs):
                            self.observation['data'].update(kwargs)
                        def __enter__(self):
                            return self
                        def __exit__(self, *args):
                            pass
                    return ObservationWrapper(observation)

            return TraceContext(self.client, context)

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

            def _create_dummy_observation(self):
                class DummyObservation:
                    def update(self, **kwargs):
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

            if hasattr(trace, observation_type):
                # Use the trace wrapper methods
                observation = getattr(trace, observation_type)(**observation_data)
            else:
                # Fallback to span if observation type not found
                observation = trace.span(**observation_data)

            return observation
        except Exception as e:
            print(f"Failed to add observation: {e}")
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
        return self.add_observation(
            trace=trace,
            name=f"{agent_name}_execution",
            observation_type="span",
            input=input_data,
            output=output_data,
            metadata={
                "agent_name": agent_name,
                "agent_type": agent_type,
                "execution_time_seconds": execution_time,
                "timestamp": datetime.utcnow().isoformat(),
                **(metadata or {})
            }
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
        return self.add_observation(
            trace=trace,
            name="intent_classification",
            observation_type="event",
            input=query,
            output=predicted_class,
            metadata={
                "confidence": confidence,
                "all_scores": all_scores,
                "threshold": settings.confidence_threshold,
                "classification_method": "semantic_similarity"
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
        return self.add_observation(
            trace=trace,
            name="rag_retrieval",
            observation_type="span",
            input=query,
            output=f"Retrieved {len(retrieved_docs)} documents",
            metadata={
                "num_documents": len(retrieved_docs),
                "similarity_scores": similarity_scores,
                "average_similarity": sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0,
                "retrieval_time_seconds": retrieval_time,
                "embedding_model": settings.embedding_model
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
        return self.add_observation(
            trace=trace,
            name="quality_evaluation",
            observation_type="generation",
            input=query,
            output=answer,
            metadata={
                "context_length": len(context),
                "answer_length": len(answer),
                "quality_scores": quality_scores,
                "overall_score": overall_score,
                "evaluation_dimensions": settings.quality_dimensions,
                "evaluator_model": settings.evaluator_model
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
            input=str(context) if context else "",
            output=error_message,
            metadata={
                "error_type": error_type,
                "timestamp": datetime.utcnow().isoformat(),
                **(context or {})
            }
        )

    def flush(self) -> None:
        """Flush any pending traces to Langfuse."""
        if self.enabled and self.client:
            try:
                self.client.flush()
            except Exception as e:
                print(f"Failed to flush Langfuse: {e}")


# Global Langfuse client instance
langfuse_client = LangfuseClient()