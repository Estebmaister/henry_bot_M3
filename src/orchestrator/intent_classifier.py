"""
Intent classification for routing queries to appropriate department agents.
"""

import asyncio
from typing import Dict, Tuple, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from langfuse import observe

from src.config import settings
from src.utils import langfuse_client


class IntentClassifier:
    """
    Semantic intent classifier for routing queries to department-specific agents.
    Uses embedding similarity to classify queries into HR, Tech, or Finance categories.
    """

    def __init__(self, embedding_model: str = None):
        """
        Initialize the intent classifier.

        Args:
            embedding_model: Name of the embedding model to use
        """
        self.embedding_model = embedding_model or settings.embedding_model
        self._embeddings = None
        self._department_embeddings = {}
        self._initialized = False

        # Department-specific query examples for semantic anchoring
        self.department_examples = {
            'hr': [
                "What benefits am I entitled to as a new employee?",
                "How do I request time off for vacation?",
                "What's the company policy on remote work?",
                "How do I enroll in the health insurance plan?",
                "What is the dress code policy?",
                "How do I report workplace harassment?",
                "When is open enrollment for benefits?",
                "What's the parental leave policy?",
                "How do I update my personal information?",
                "What training opportunities are available?"
            ],
            'tech': [
                "My laptop won't connect to the VPN",
                "How do I reset my password for the internal systems?",
                "What's the process for getting software development tools installed?",
                "How do I report a security vulnerability?",
                "Why is the company VPN so slow?",
                "How do I get access to the production database?",
                "How do I connect to the office Wi-Fi?",
                "Can I get a second monitor for my home setup?",
                "How do I install approved software?",
                "What should I do if my computer is running slow?"
            ],
            'finance': [
                "What's the procedure for requesting a salary advance?",
                "How do I submit expense reports for reimbursement?",
                "What's our department budget for Q3?",
                "How do I get approval for a software purchase?",
                "When do we get paid?",
                "How do I use the company credit card?",
                "What's the process for vendor payment?",
                "How do I request budget approval?",
                "What expenses are reimbursable?",
                "How do I submit my timesheet?"
            ]
        }

    async def initialize(self) -> None:
        """
        Initialize the classifier by pre-computing department embeddings.
        """
        try:
            print("Initializing intent classifier...")
            self._embeddings = SentenceTransformer(self.embedding_model)

            # Compute embeddings for each department's example queries
            for department, examples in self.department_examples.items():
                dept_embeddings = self._embeddings.encode(
                    examples,
                    batch_size=8,
                    show_progress_bar=False
                )
                # Store the mean embedding as the department centroid
                self._department_embeddings[department] = np.mean(dept_embeddings, axis=0)

            self._initialized = True
            print(f"Intent classifier initialized for {len(self._department_embeddings)} departments")

        except Exception as e:
            print(f"Error initializing intent classifier: {e}")
            raise

    @observe(name="intent_classification")
    async def classify(self, query: str, trace=None) -> Tuple[str, float, Dict[str, float]]:
        """
        Classify the query intent and return the predicted department.
        @observe decorator automatically creates a nested observation.

        Args:
            query: The input query to classify
            trace: Optional Langfuse trace for backward compatibility

        Returns:
            Tuple of (predicted_department, confidence, all_scores)
        """
        if not self._initialized:
            raise RuntimeError("Intent classifier not initialized. Call initialize() first.")

        try:
            # Generate query embedding
            query_embedding = self._embeddings.encode([query])[0]

            # Calculate similarity with each department
            similarities = {}
            for department, dept_embedding in self._department_embeddings.items():
                # Cosine similarity
                similarity = np.dot(query_embedding, dept_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(dept_embedding)
                )
                similarities[department] = float(similarity)

            # Find the best matching department
            predicted_department = max(similarities, key=similarities.get)
            confidence = similarities[predicted_department]

            # Apply confidence threshold
            if confidence < settings.confidence_threshold:
                predicted_department = "general"  # Fallback to general category
                confidence = max(similarities.values())  # Still return the highest confidence

            # Log classification with Langfuse
            if trace:
                langfuse_client.log_classification_result(
                    trace=trace,
                    query=query,
                    predicted_class=predicted_department,
                    confidence=confidence,
                    all_scores=similarities
                )

            return predicted_department, confidence, similarities

        except Exception as e:
            error_msg = f"Error during intent classification: {e}"

            # Log error with Langfuse
            if trace:
                langfuse_client.log_error(
                    trace=trace,
                    error_message=error_msg,
                    error_type="intent_classification_error",
                    context={'query': query}
                )

            # Return fallback classification
            return "general", 0.0, {dept: 0.0 for dept in self.department_examples.keys()}

    def is_initialized(self) -> bool:
        """
        Check if the classifier is initialized.

        Returns:
            True if initialized, False otherwise
        """
        return self._initialized