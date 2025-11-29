"""
Main orchestrator agent for multi-agent coordination and routing.
"""

import asyncio
import time
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

from .intent_classifier import IntentClassifier
from ..agents import RAGAgent
from ..config import settings
from ..utils import langfuse_client
from ..agents import AgentResponse
from langfuse import observe


@dataclass
class OrchestratorResponse:
    """Dataclass for orchestrator responses."""
    answer: str
    department: str
    confidence: float
    agent_used: str
    processing_time: float
    source_documents: list
    classification_confidence: float
    metadata: dict


class MultiAgentOrchestrator:
    """
    Main orchestrator that coordinates multiple specialized agents.
    Handles intent classification, routing, and response aggregation.
    """

    def __init__(self, force_rebuild: bool = False, use_persistent: bool = True):
        """Initialize the multi-agent orchestrator.

        Args:
            force_rebuild: Force rebuild all FAISS indices
            use_persistent: Enable/disable persistent storage
        """
        self.intent_classifier = IntentClassifier()
        self.agents: Dict[str, RAGAgent] = {}
        self._initialized = False
        self._force_rebuild = force_rebuild
        self._use_persistent = use_persistent

        # Define agent configurations
        self.agent_configs = {
            'hr': {
                'name': 'HR Assistant',
                'documents_path': settings.hr_docs_dir,
                'department': 'hr'
            },
            'tech': {
                'name': 'IT Support Assistant',
                'documents_path': settings.tech_docs_dir,
                'department': 'tech'
            },
            'finance': {
                'name': 'Finance Assistant',
                'documents_path': settings.finance_docs_dir,
                'department': 'finance'
            }
        }

    async def initialize(self) -> None:
        """
        Initialize the orchestrator and all agents.
        """
        try:
            print("Initializing Multi-Agent Orchestrator...")

            # Initialize intent classifier
            await self.intent_classifier.initialize()

            # Initialize all department agents
            initialization_tasks = []
            for dept, config in self.agent_configs.items():
                agent = RAGAgent(
                    name=config['name'],
                    department=config['department'],
                    documents_path=config['documents_path']
                )
                # Pass configuration to agent's retriever
                agent._retriever.force_rebuild = self._force_rebuild
                agent._retriever.use_persistent_storage = self._use_persistent

                self.agents[dept] = agent
                initialization_tasks.append(agent.initialize())

            # Run all agent initializations concurrently
            await asyncio.gather(*initialization_tasks, return_exceptions=True)

            self._initialized = True
            print("Multi-Agent Orchestrator initialized successfully")

        except Exception as e:
            print(f"Error initializing orchestrator: {e}")
            raise

    @observe(name="multi_agent_query_processing")
    async def process_query(self, query: str, user_id: Optional[str] = None) -> OrchestratorResponse:
        """
        Process a query through the complete multi-agent pipeline.
        @observe decorator automatically creates a trace and links all nested calls.

        Args:
            query: The input query to process
            user_id: Optional user identifier for tracing

        Returns:
            OrchestratorResponse with complete results
        """
        if not self._initialized:
            raise RuntimeError(
                "Orchestrator not initialized. Call initialize() first.")

        # The @observe decorator above automatically creates the trace
        # Add metadata to the current trace context
        if hasattr(langfuse_client.client, 'update_current_trace'):
            langfuse_client.client.update_current_trace(
                metadata={
                    "query_length": len(query),
                    "service_name": "henry_bot_M3",
                    "source": "terminal_cli",
                    "user_id": user_id
                }
            )

        # Use dummy trace context for backward compatibility with existing code
        trace = langfuse_client._create_dummy_trace()

        start_time = time.time()

        try:
            # Create comprehensive workflow span
            with trace.span(
                name="orchestrator_workflow",
                input=query,
                metadata={
                    'stage': 'complete_orchestration',
                    'service_name': 'henry_bot_M3',
                    'user_id': user_id or 'terminal'
                }
            ) as workflow_span:

                # Step 1: Intent Classification with span
                with trace.span(
                    name="intent_classification",
                    input=query,
                    metadata={
                        'stage': 'classification',
                        'available_departments': list(self.agents.keys())
                    }
                ) as classification_span:

                    classification_start = time.time()
                    predicted_department, classification_confidence, all_scores = await self.intent_classifier.classify(query, trace)
                    classification_time = time.time() - classification_start

                    # Update classification span with results
                    classification_span.update(
                        output=predicted_department,
                        metadata={
                            'predicted_department': predicted_department,
                            'confidence': classification_confidence,
                            'classification_time_seconds': classification_time,
                            'all_scores': all_scores,
                            'threshold_met': classification_confidence > settings.confidence_threshold
                        }
                    )

                    # Also log as event for additional visibility
                    langfuse_client.log_classification_result(
                        trace=trace,
                        query=query,
                        predicted_class=predicted_department,
                        confidence=classification_confidence,
                        all_scores=all_scores
                    )

                # Step 2: Agent Processing with span
                with trace.span(
                    name="agent_processing",
                    input=f"Query: {query[:100]}... -> Department: {predicted_department}",
                    metadata={
                        'stage': 'agent_execution',
                        'target_department': predicted_department,
                        'agent_available': predicted_department in self.agents
                    }
                ) as agent_span:

                    agent_start = time.time()
                    if predicted_department not in self.agents:
                        # Fallback handling for unknown departments
                        with trace.span(
                            name="fallback_handling",
                            metadata={
                                'stage': 'fallback',
                                'reason': 'department_not_available',
                                'requested_department': predicted_department
                            }
                        ) as fallback_span:

                            response = await self._handle_fallback(query, predicted_department, trace)
                            agent_execution_time = time.time() - agent_start

                            fallback_span.update(
                                output=response.answer,
                                metadata={
                                    'fallback_used': True,
                                    'fallback_reason': 'department_not_available'
                                }
                            )
                    else:
                        # Normal agent processing
                        with trace.span(
                            name="rag_agent_execution",
                            metadata={
                                'stage': 'rag_processing',
                                'agent_name': f"{predicted_department}_assistant",
                                'agent_type': 'rag'
                            }
                        ) as rag_span:

                            response = await self._route_to_agent(query, predicted_department, trace)
                            agent_execution_time = time.time() - agent_start

                            rag_span.update(
                                output=response.answer[:200] + "..." if len(
                                    response.answer) > 200 else response.answer,
                                metadata={
                                    'department': predicted_department,
                                    'num_source_documents': len(response.source_documents),
                                    'agent_confidence': response.confidence,
                                    'agent_execution_time_seconds': agent_execution_time
                                }
                            )

                    # Update agent processing span
                    agent_span.update(
                        output=response.answer[:500] + "..." if len(
                            response.answer) > 500 else response.answer,
                        metadata={
                            'agent_execution_time_seconds': agent_execution_time,
                            'agent_used': response.metadata.get('agent_name', 'unknown'),
                            'fallback_used': predicted_department not in self.agents
                        }
                    )

                    # Also log as event for metrics tracking
                    langfuse_client.log_agent_execution(
                        trace=trace,
                        agent_name=response.metadata.get(
                            'agent_name', predicted_department),
                        agent_type="rag_agent",
                        input_data=query,
                        output_data=response.answer,
                        execution_time=agent_execution_time,
                        metadata={
                            'department': predicted_department,
                            'num_source_documents': len(response.source_documents),
                            'agent_confidence': response.confidence,
                            'documents_retrieved': len(response.source_documents)
                        }
                    )

            # Calculate total processing time
            total_time = time.time() - start_time

            # Create orchestrator response
            orchestrator_response = OrchestratorResponse(
                answer=response.answer,
                department=predicted_department,
                confidence=response.confidence,
                agent_used=response.metadata.get('agent_name', 'unknown'),
                processing_time=total_time,
                source_documents=response.source_documents,
                classification_confidence=classification_confidence,
                metadata={
                    **response.metadata,
                    'classification_time': classification_time,
                    'agent_execution_time': agent_execution_time,
                    'all_classification_scores': all_scores,
                    'predicted_department': predicted_department,
                    'fallback_used': predicted_department not in self.agents
                }
            )

            # Update trace with final output and comprehensive metadata
            if trace:
                trace.update(
                    output=orchestrator_response.answer,
                    metadata={
                        'department': orchestrator_response.department,
                        'confidence': orchestrator_response.confidence,
                        'processing_time': orchestrator_response.processing_time,
                        'agent_used': orchestrator_response.agent_used,
                        'classification_confidence': classification_confidence,
                        'num_source_documents': len(orchestrator_response.source_documents),
                        'fallback_used': orchestrator_response.metadata.get('fallback_used', False),
                        'classification_time': classification_time,
                        'agent_execution_time': agent_execution_time
                    }
                )

            return orchestrator_response

        except Exception as e:
            error_time = time.time() - start_time
            error_msg = f"Error processing query: {e}"

            # Log error with Langfuse
            if trace:
                langfuse_client.log_error(
                    trace=trace,
                    error_message=error_msg,
                    error_type="orchestrator_processing_error",
                    context={
                        'query': query,
                        'processing_time': error_time
                    }
                )

            # Return error response
            return OrchestratorResponse(
                answer="I apologize, but I encountered an error while processing your request. Please try again or contact support if the issue persists.",
                department="error",
                confidence=0.0,
                agent_used="none",
                processing_time=error_time,
                source_documents=[],
                classification_confidence=0.0,
                metadata={'error': str(e), 'fallback_used': True}
            )

        finally:
            # Note: Trace flushing is handled by the main system to ensure complete operation timing
            pass

    async def _route_to_agent(self, query: str, department: str, trace) -> any:
        """
        Route query to the appropriate department agent.

        Args:
            query: The input query
            department: Target department
            trace: Langfuse trace

        Returns:
            Agent response
        """
        agent = self.agents[department]
        return await agent.process_query(query, trace)

    async def _handle_fallback(self, query: str, predicted_department: str, trace) -> any:
        """
        Handle queries for departments without specialized agents.

        Args:
            query: The input query
            predicted_department: Predicted department (without agent)
            trace: Langfuse trace

        Returns:
            Fallback response
        """
        fallback_responses = {
            'general': "I understand you have a question, but I need to route you to the appropriate specialist. Could you please clarify if your question is related to HR (benefits, policies, time off), IT/Technology (software, hardware, access), or Finance (expenses, budget, purchases)?",
            'unknown': "I'm having trouble categorizing your question. Please rephrase it or let me know if it's related to HR, IT, or Finance so I can provide you with the best assistance."
        }

        fallback_answer = fallback_responses.get(
            predicted_department,
            fallback_responses['general']
        )

        # Log fallback usage
        if trace:
            langfuse_client.log_agent_execution(
                trace=trace,
                agent_name="fallback_handler",
                agent_type="fallback",
                input_data=query,
                output_data=fallback_answer,
                execution_time=0.1,
                metadata={
                    'predicted_department': predicted_department,
                    'fallback_reason': 'no_agent_available'
                }
            )

        # Create fallback response object similar to AgentResponse
        return AgentResponse(
            answer=fallback_answer,
            confidence=0.3,
            source_documents=[],
            metadata={
                'agent_name': 'fallback_handler',
                'department': predicted_department,
                'fallback_used': True
            }
        )

    def is_initialized(self) -> bool:
        """
        Check if the orchestrator is initialized.

        Returns:
            True if initialized, False otherwise
        """
        return self._initialized

    def get_available_departments(self) -> list:
        """
        Get list of available departments.

        Returns:
            List of department names
        """
        return list(self.agents.keys())

    async def health_check(self) -> Dict[str, any]:
        """
        Perform health check on all components.

        Returns:
            Health check results
        """
        if not self._initialized:
            return {'status': 'not_initialized', 'components': {}}

        health_results = {
            'status': 'healthy',
            'components': {
                'intent_classifier': self.intent_classifier.is_initialized(),
                'agents': {}
            }
        }

        # Check each agent
        for dept, agent in self.agents.items():
            health_results['components']['agents'][dept] = agent.is_initialized()

        # Overall status
        all_healthy = all([
            health_results['components']['intent_classifier'],
            all(health_results['components']['agents'].values())
        ])

        health_results['status'] = 'healthy' if all_healthy else 'degraded'
        return health_results
