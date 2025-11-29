#!/usr/bin/env python3
"""
Test suite for Langfuse client functionality.
Basic tests for client initialization and method availability.
"""

import sys
import os
import unittest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.utils.langfuse_client import LangfuseClient


class TestLangfuseClient(unittest.TestCase):
    """Test Langfuse client functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.client = LangfuseClient()

    def test_client_initialization(self):
        """Test that Langfuse client initializes properly."""
        self.assertIsNotNone(self.client)
        self.assertIsInstance(self.client, LangfuseClient)

    def test_client_methods(self):
        """Test that expected methods exist on the client."""
        expected_methods = [
            'create_trace',
            'flush',
            'log_agent_execution',
            'log_classification_result',
            'log_llm_call',
            'log_quality_evaluation'
        ]

        for method_name in expected_methods:
            with self.subTest(method=method_name):
                self.assertTrue(hasattr(self.client, method_name),
                              f"Method {method_name} should exist")

    def test_flush_method_availability(self):
        """Test that flush method exists and can be called."""
        self.assertTrue(hasattr(self.client, 'flush'),
                        "Langfuse client should have flush method")

        # Try to call flush (may fail in test environment, but method should exist)
        try:
            self.client.flush()
            print("✅ Flush method executed successfully")
        except Exception as e:
            # Expected in test environment without proper credentials
            self.assertIn("credentials", str(e).lower() or
                          "not found" in str(e).lower() or
                          "disabled" in str(e).lower(),
                          f"Expected credential-related error, got: {e}")

    def test_create_trace_method(self):
        """Test create_trace method exists and returns expected interface."""
        self.assertTrue(hasattr(self.client, 'create_trace'))

        # Test that it returns a trace context (even in dummy mode)
        trace = self.client.create_trace(
            name="test_trace",
            input="test input",
            user_id="test_user"
        )

        self.assertIsNotNone(trace)
        # The trace should have span, generation, event methods
        self.assertTrue(hasattr(trace, 'span'))
        self.assertTrue(hasattr(trace, 'generation'))
        self.assertTrue(hasattr(trace, 'event'))

    def test_dummy_trace_interface(self):
        """Test that dummy trace provides proper interface."""
        trace = self.client.create_trace(
            name="test_trace",
            input="test input"
        )

        # Test context manager interface
        with trace.span("test_span") as span:
            self.assertIsNotNone(span)

        with trace.generation("test_generation") as gen:
            self.assertIsNotNone(gen)

        with trace.event("test_event") as event:
            self.assertIsNotNone(event)


if __name__ == "__main__":
    unittest.main()