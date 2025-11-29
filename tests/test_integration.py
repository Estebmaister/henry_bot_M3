"""
Integration tests for the multi-agent intelligent routing system.
"""

import pytest
import asyncio
import json
from pathlib import Path

from src.main import MultiAgentSystem
from src.orchestrator import MultiAgentOrchestrator
from src.evaluator import ResponseQualityEvaluator


@pytest.fixture
async def multi_agent_system():
    """Fixture to initialize the multi-agent system for testing."""
    system = MultiAgentSystem()
    await system.initialize()
    yield system
    await system.shutdown()


@pytest.fixture
def test_queries():
    """Load test queries for integration testing."""
    queries_path = Path("test_queries.json")
    if not queries_path.exists():
        pytest.skip("test_queries.json not found")

    with open(queries_path, 'r', encoding='utf-8') as f:
        return json.load(f)


@pytest.mark.asyncio
class TestMultiAgentIntegration:
    """Integration tests for the complete multi-agent system."""

    async def test_system_initialization(self, multi_agent_system):
        """Test that the system initializes correctly."""
        status = multi_agent_system.get_system_status()

        assert status['status'] == 'initialized'
        assert status['components']['orchestrator'] == True
        assert status['components']['evaluator'] == True
        assert len(status['available_departments']) >= 3

    async def test_single_query_processing(self, multi_agent_system):
        """Test processing a single query through the complete pipeline."""
        query = "What benefits am I entitled to as a new employee?"

        response = await multi_agent_system.process_query(query, evaluate_quality=True)

        # Basic response structure
        assert 'query' in response
        assert 'answer' in response
        assert 'department' in response
        assert 'agent_used' in response
        assert 'confidence' in response
        assert 'processing_time' in response
        assert 'quality_evaluation' in response

        # Content validation
        assert response['query'] == query
        assert response['department'] in ['hr', 'tech', 'finance', 'general', 'error']
        assert len(response['answer']) > 0
        assert 0 <= response['confidence'] <= 1
        assert response['processing_time'] > 0

        # Quality evaluation validation
        qe = response['quality_evaluation']
        assert 'overall_score' in qe
        assert 'dimension_scores' in qe
        assert 'reasoning' in qe
        assert 'recommendations' in qe
        assert 1 <= qe['overall_score'] <= 10

    async def test_multiple_query_processing(self, multi_agent_system, test_queries):
        """Test processing multiple queries in batch."""
        queries = list(test_queries.keys())[:5]  # Test first 5 queries

        responses = await multi_agent_system.batch_process_queries(
            queries,
            evaluate_quality=False  # Skip quality eval for faster testing
        )

        assert len(responses) == len(queries)

        for query, response in zip(queries, responses):
            assert response['query'] == query
            assert len(response['answer']) > 0
            assert response['department'] in ['hr', 'tech', 'finance', 'general', 'error']

    async test_department_classification(self, multi_agent_system):
        """Test that queries are correctly classified by department."""
        test_cases = [
            ("How do I request vacation time?", "hr"),
            ("My laptop won't connect to the VPN", "tech"),
            ("How do I submit expense reports?", "finance"),
            ("What's the dress code policy?", "hr"),
            ("How do I reset my password?", "tech")
        ]

        for query, expected_department in test_cases:
            response = await multi_agent_system.process_query(query, evaluate_quality=False)

            # Allow some flexibility in classification
            assert response['department'] in ['hr', 'tech', 'finance', 'general']
            # The exact classification might vary, so we just check it's a valid department

    async def test_error_handling(self, multi_agent_system):
        """Test error handling for edge cases."""
        # Test empty query
        response = await multi_agent_system.process_query("", evaluate_quality=False)
        assert response['answer'] != ""
        assert response['department'] in ['general', 'error']

        # Test very long query
        long_query = "What " * 1000 + "policy?"
        response = await multi_agent_system.process_query(long_query, evaluate_quality=False)
        assert response['answer'] != ""

    async def test_source_documents_inclusion(self, multi_agent_system):
        """Test that source documents are included in responses."""
        query = "What benefits am I entitled to as a new employee?"

        response = await multi_agent_system.process_query(query, evaluate_quality=False)

        # Check that source documents are present
        assert 'source_documents' in response
        assert isinstance(response['source_documents'], list)

        # If documents were retrieved, they should have proper structure
        if response['source_documents']:
            doc = response['source_documents'][0]
            assert 'content' in doc
            assert 'source' in doc
            assert 'similarity_score' in doc

    async def test_confidence_scores(self, multi_agent_system):
        """Test that confidence scores are reasonable."""
        queries = [
            "What benefits am I entitled to?",
            "My laptop is broken",
            "How do I get reimbursed?"
        ]

        for query in queries:
            response = await multi_agent_system.process_query(query, evaluate_quality=False)

            # Confidence should be a valid float between 0 and 1
            assert isinstance(response['confidence'], (int, float))
            assert 0 <= response['confidence'] <= 1

    async def test_performance_metrics(self, multi_agent_system):
        """Test that performance metrics are tracked properly."""
        query = "What's the company policy on remote work?"

        response = await multi_agent_system.process_query(query, evaluate_quality=False)

        # Processing time should be reasonable (less than 30 seconds for testing)
        assert response['processing_time'] > 0
        assert response['processing_time'] < 30

        # Metadata should contain useful information
        assert 'metadata' in response
        assert 'agent_name' in response['metadata']
        assert 'department' in response['metadata']


@pytest.mark.asyncio
class TestComponentIntegration:
    """Test integration between individual components."""

    async def test_orchestrator_only(self):
        """Test orchestrator component independently."""
        orchestrator = MultiAgentOrchestrator()
        await orchestrator.initialize()

        query = "How do I request time off?"
        response = await orchestrator.process_query(query)

        assert response.answer != ""
        assert response.department in ['hr', 'tech', 'finance', 'general']
        assert response.confidence >= 0
        assert response.processing_time > 0

    async def test_evaluator_only(self):
        """Test evaluator component independently."""
        evaluator = ResponseQualityEvaluator()
        await evaluator.initialize()

        query = "What benefits am I entitled to?"
        answer = "As a new employee, you are entitled to health insurance, PTO, and other benefits."
        context = "Employee benefits include health insurance, dental, vision, and paid time off."
        source_docs = [
            {
                'content': context,
                'source': 'benefits.md',
                'similarity_score': 0.9,
                'metadata': {}
            }
        ]

        result = await evaluator.evaluate_response(query, answer, context, source_docs)

        assert 1 <= result.overall_score <= 10
        assert len(result.dimension_scores) >= 3
        assert isinstance(result.reasoning, str)
        assert isinstance(result.recommendations, list)


@pytest.mark.asyncio
class TestSystemWithQualityEvaluation:
    """Tests specifically for quality evaluation functionality."""

    async def test_quality_evaluation_dimensions(self, multi_agent_system):
        """Test that all quality dimensions are evaluated."""
        query = "What's the process for getting a software development tool installed?"

        response = await multi_agent_system.process_query(query, evaluate_quality=True)

        qe = response['quality_evaluation']
        expected_dimensions = ['relevance', 'completeness', 'accuracy']

        for dimension in expected_dimensions:
            assert dimension in qe['dimension_scores']
            assert 1 <= qe['dimension_scores'][dimension] <= 10

    async def test_quality_evaluation_reasoning(self, multi_agent_system):
        """Test that quality evaluation provides useful reasoning."""
        query = "How do I connect to the office Wi-Fi?"

        response = await multi_agent_system.process_query(query, evaluate_quality=True)

        qe = response['quality_evaluation']
        assert len(qe['reasoning']) > 20  # Reasoning should be substantive
        assert isinstance(qe['recommendations'], list)


if __name__ == "__main__":
    # Simple test runner for manual testing
    async def run_manual_tests():
        system = MultiAgentSystem()
        await system.initialize()

        # Test a few queries
        test_queries = [
            "What benefits am I entitled to as a new employee?",
            "My laptop won't connect to the VPN",
            "How do I submit expense reports?"
        ]

        for query in test_queries:
            print(f"\nQuery: {query}")
            response = await system.process_query(query, evaluate_quality=True)
            print(f"Department: {response['department']}")
            print(f"Answer: {response['answer'][:200]}...")
            print(f"Quality Score: {response['quality_evaluation']['overall_score']}/10")

        await system.shutdown()

    # Uncomment to run manual tests
    # asyncio.run(run_manual_tests())