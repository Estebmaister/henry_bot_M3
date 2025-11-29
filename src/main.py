"""
Main entry point for the Multi-Agent Intelligent Routing System - Fixed Version.

This module provides the primary interface for the multi-agent system,
handling initialization, query processing, and quality evaluation.
"""

import os
# Set tokenizer parallelism early to avoid warnings
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import asyncio
import json
import time
from typing import Dict, List, Optional, Any
from pathlib import Path

from src.orchestrator import MultiAgentOrchestrator
from src.evaluator import ResponseQualityEvaluator
from src.config import settings
from src.utils import langfuse_client
from src.retrievers.cached_faiss_retriever import CachedFAISSRetriever


class MultiAgentSystem:
    """
    Main system that orchestrates query processing, routing, and quality evaluation.
    Provides a unified interface for the complete multi-agent workflow.
    """

    def __init__(self, force_rebuild: bool = False, use_persistent: bool = True):
        """Initialize the multi-agent system.

        Args:
            force_rebuild: Force rebuild all FAISS indices
            use_persistent: Enable/disable persistent storage
        """
        self.orchestrator = MultiAgentOrchestrator(force_rebuild=force_rebuild, use_persistent=use_persistent)
        self.evaluator = ResponseQualityEvaluator()
        self._initialized = False
        self.force_rebuild = force_rebuild
        self.use_persistent = use_persistent

        # Store configuration for agents
        self._force_rebuild = force_rebuild
        self._use_persistent = use_persistent

        if force_rebuild:
            print("Force rebuild mode enabled")
        if not use_persistent:
            print("Persistent storage disabled")

    async def initialize(self) -> None:
        """
        Initialize the complete multi-agent system.
        """
        try:
            print("🚀 Initializing Multi-Agent Intelligent Routing System...")
            start_time = time.time()

            # Initialize orchestrator (which initializes all agents)
            await self.orchestrator.initialize()

            # Initialize quality evaluator
            await self.evaluator.initialize()

            initialization_time = time.time() - start_time
            self._initialized = True

            print(f"✅ Multi-Agent System initialized successfully in {initialization_time:.2f} seconds")
            print("System initialized successfully")

        except Exception as e:
            print(f"Error initializing system: {e}")
            raise

    async def process_query(
        self,
        query: str,
        user_id: Optional[str] = None,
        evaluate_quality: bool = True
    ) -> Dict[str, Any]:
        """
        Process a query through the complete multi-agent pipeline.

        Args:
            query: The query to process
            user_id: Optional user ID for tracing
            evaluate_quality: Whether to run quality evaluation

        Returns:
            Dictionary containing the complete response
        """
        if not self._initialized:
            raise RuntimeError("System not initialized. Call initialize() first.")

        try:
            # Process query through orchestrator
            response = await self.orchestrator.process_query(query, user_id)

            # Convert to dictionary format
            result = {
                'query': query,
                'answer': response.answer,
                'department': response.department,
                'agent_used': response.agent_used,
                'confidence': response.confidence,
                'classification_confidence': response.classification_confidence,
                'processing_time': response.processing_time,
                'source_documents': response.source_documents,  # Already in the correct format
                'metadata': response.metadata
            }

            # Run quality evaluation if requested
            if evaluate_quality:
                try:
                    # Combine source documents for context
                    context = "\n\n".join([
                        doc['content'] for doc in result['source_documents']
                    ])

                    evaluation = await self.evaluator.evaluate_response(
                        query=result['query'],
                        answer=result['answer'],
                        context=context,
                        source_documents=result['source_documents']
                    )

                    result['quality_evaluation'] = {
                        'overall_score': evaluation.overall_score,
                        'dimension_scores': evaluation.dimension_scores,
                        'reasoning': evaluation.reasoning,
                        'recommendations': evaluation.recommendations
                    }

                except Exception as e:
                    print(f"Quality evaluation failed: {e}")
                    result['quality_evaluation'] = {
                        'overall_score': 5.0,
                        'dimension_scores': {'relevance': 5.0, 'completeness': 5.0, 'accuracy': 5.0},
                        'reasoning': 'Evaluation failed',
                        'recommendations': ['Manual review recommended']
                    }

            return result

        except Exception as e:
            print(f"Error processing query: {e}")
            return {
                'query': query,
                'answer': "I apologize, but I encountered an error while processing your request. Please try again or contact support if the issue persists.",
                'department': 'error',
                'agent_used': 'none',
                'confidence': 0.0,
                'processing_time': 0.0,
                'source_documents': [],
                'metadata': {'error': str(e)}
            }

    async def test_with_queries(
        self,
        test_file: str,
        evaluate_quality: bool = True
    ) -> Dict[str, Any]:
        """
        Test the system with a set of predefined queries.

        Args:
            test_file: Path to JSON file containing test queries
            evaluate_quality: Whether to run quality evaluation

        Returns:
            Dictionary containing test results
        """
        try:
            # Load test queries
            with open(test_file, 'r', encoding='utf-8') as f:
                test_data = json.load(f)

            queries = test_data.get('queries', [])
            results = []

            print(f"Running {len(queries)} test queries...")
            start_time = time.time()

            # Process each query
            for i, query_data in enumerate(queries):
                query = query_data.get('query', '')
                expected_department = query_data.get('expected_department')
                user_id = query_data.get('user_id', f'test_user_{i}')

                print(f"\nProcessing query {i + 1}/{len(queries)}: {query[:50]}...")

                result = await self.process_query(query, user_id, evaluate_quality)
                results.append({
                    'query': query,
                    'expected_department': expected_department,
                    'result': result
                })

            # Calculate statistics
            total_time = time.time() - start_time
            correct_classifications = sum(
                1 for r in results
                if r['expected_department'] and r['result']['department'] == r['expected_department']
            )

            classification_accuracy = correct_classifications / len(results) if results else 0
            avg_confidence = sum(r['result']['confidence'] for r in results) / len(results) if results else 0
            avg_processing_time = sum(r['result']['processing_time'] for r in results) / len(results) if results else 0

            if evaluate_quality:
                avg_quality_score = sum(
                    r['result'].get('quality_evaluation', {}).get('overall_score', 0)
                    for r in results
                ) / len(results) if results else 0
            else:
                avg_quality_score = None

            return {
                'test_results': results,
                'test_summary': {
                    'total_queries': len(results),
                    'classification_accuracy': classification_accuracy,
                    'avg_confidence': avg_confidence,
                    'avg_processing_time': avg_processing_time,
                    'avg_quality_score': avg_quality_score,
                    'total_time': total_time
                }
            }

        except FileNotFoundError:
            return {'error': f'Test file not found: {test_file}'}
        except json.JSONDecodeError:
            return {'error': f'Invalid JSON in test file: {test_file}'}
        except Exception as e:
            return {'error': f'Error running tests: {e}'}

    def get_system_status(self) -> Dict[str, Any]:
        """
        Get the current status of the system.

        Returns:
            Dictionary containing system status information
        """
        if not self._initialized:
            return {
                'status': 'not_initialized',
                'components': {
                    'orchestrator': False,
                    'evaluator': False
                }
            }

        # Get health check from orchestrator
        try:
            health = self.orchestrator.health_check()
            return {
                'status': health['status'],
                'components': {
                    'orchestrator': self.orchestrator.is_initialized(),
                    'evaluator': self.evaluator.is_initialized(),
                    **health.get('components', {})
                },
                'available_departments': self.orchestrator.get_available_departments(),
                'config': {
                    'model_name': settings.model_name,
                    'embedding_model': settings.embedding_model,
                    'similarity_top_k': settings.similarity_top_k,
                    'confidence_threshold': settings.confidence_threshold
                }
            }

        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'components': {
                    'orchestrator': self.orchestrator.is_initialized(),
                    'evaluator': self.evaluator.is_initialized()
                }
            }

    async def shutdown(self) -> None:
        """
        Shutdown the system and clean up resources.
        """
        try:
            print("Shutting down Multi-Agent System...")

            # Flush any pending Langfuse traces
            langfuse_client.flush()

            self._initialized = False
            print("Multi-Agent System shutdown complete")

        except Exception as e:
            print(f"Error during shutdown: {e}")

    def clear_cache(self, department: str = None) -> None:
        """Clear system caches.

        Args:
            department: Specific department cache to clear (optional)
        """
        cache_dir = Path(settings.cache_dir)
        if department:
            # Clear specific department cache
            dept_cache_dir = cache_dir / department
            if dept_cache_dir.exists():
                import shutil
                shutil.rmtree(dept_cache_dir)
                print(f"Cleared cache for department: {department}")
            else:
                print(f"No cache found for department: {department}")
        else:
            # Clear all caches
            CachedFAISSRetriever.clear_all_caches()
            print("All system caches cleared")

    def clear_store(self, department: str = None) -> None:
        """Clear persistent FAISS indices storage.

        Args:
            department: Specific department store to clear (optional)
        """
        if department:
            # Clear specific department store
            store_dirs = [
                Path(settings.faiss_indices_dir) / department,
                Path(settings.embeddings_dir) / department,
                Path(settings.metadata_dir) / department
            ]
            cleared_any = False
            for store_dir in store_dirs:
                if store_dir.exists():
                    import shutil
                    shutil.rmtree(store_dir)
                    print(f"Cleared store for {department}: {store_dir}")
                    cleared_any = True
            if not cleared_any:
                print(f"No store found for department: {department}")
        else:
            # Clear all stores
            store_dirs = [Path(settings.faiss_indices_dir), Path(settings.embeddings_dir), Path(settings.metadata_dir)]
            for store_dir in store_dirs:
                if store_dir.exists():
                    import shutil
                    shutil.rmtree(store_dir)
                    print(f"Cleared store directory: {store_dir}")
            print("All persistent stores cleared")

    def get_store_info(self) -> Dict[str, Any]:
        """Get information about persistent FAISS indices storage."""
        # Use default paths if settings are not accessible
        faiss_indices_dir = getattr(settings, 'faiss_indices_dir', './store/faiss_indices')
        embeddings_dir = getattr(settings, 'embeddings_dir', './store/embeddings')
        metadata_dir = getattr(settings, 'metadata_dir', './store/metadata')
        use_persistent = getattr(settings, 'use_persistent_storage', True)

        store_info = {
            'store_enabled': use_persistent,
            'faiss_indices_dir': str(Path(faiss_indices_dir)),
            'embeddings_dir': str(Path(embeddings_dir)),
            'metadata_dir': str(Path(metadata_dir)),
            'departments': {}
        }

        departments = ['hr', 'tech', 'finance']
        for dept in departments:
            dept_info = {
                'faiss_index': (Path(faiss_indices_dir) / dept / "faiss.index").exists(),
                'embeddings': (Path(embeddings_dir) / dept / "embeddings.npy").exists(),
                'metadata': (Path(metadata_dir) / dept / "metadata.json").exists(),
                'documents': (Path(metadata_dir) / dept / "documents.json").exists()
            }

            # Calculate total size for this department
            total_size = 0
            for base_dir in [faiss_indices_dir, embeddings_dir, metadata_dir]:
                dept_dir = Path(base_dir) / dept
                if dept_dir.exists():
                    for file_path in dept_dir.rglob('*'):
                        if file_path.is_file():
                            total_size += file_path.stat().st_size

            dept_info['total_size_mb'] = total_size / (1024 * 1024)
            # All files except size exist
            required_files = ['faiss_index', 'embeddings', 'metadata', 'documents']
            dept_info['complete'] = all(dept_info[file] for file in required_files)

            store_info['departments'][dept] = dept_info

        return store_info

    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cached data."""
        cache_dir = Path(settings.cache_dir)
        cache_info = {
            'cache_dir': str(cache_dir),
            'exists': cache_dir.exists(),
            'departments': {}
        }

        if cache_dir.exists():
            for dept_dir in cache_dir.iterdir():
                if dept_dir.is_dir():
                    embeddings_file = dept_dir / settings.embeddings_cache_file
                    index_file = dept_dir / settings.faiss_index_file
                    metadata_file = dept_dir / settings.metadata_cache_file

                    cache_info['departments'][dept_dir.name] = {
                        'embeddings_cached': embeddings_file.exists(),
                        'index_cached': index_file.exists(),
                        'metadata_cached': metadata_file.exists(),
                        'total_size_mb': sum(
                            f.stat().st_size for f in dept_dir.rglob('*') if f.is_file()
                        ) / (1024 * 1024)
                    }

        return cache_info


# CLI interface
async def main():
    """Main CLI interface for the multi-agent system."""
    import argparse

    parser = argparse.ArgumentParser(description="Multi-Agent Intelligent Routing System")
    parser.add_argument('command', choices=['init', 'query', 'test', 'status', 'cache-clear', 'cache-info', 'store-clear', 'store-info'], help='Command to execute')
    parser.add_argument('--query', type=str, help='Query to process')
    parser.add_argument('--file', type=str, default='test_queries.json', help='Test queries file')
    parser.add_argument('--no-evaluation', action='store_true', help='Skip quality evaluation')
    parser.add_argument('--user-id', type=str, help='User ID for tracing')
    parser.add_argument('--department', type=str, help='Department for cache operations (hr, tech, finance)')
    parser.add_argument('--force-rebuild', action='store_true', help='Force rebuild all FAISS indices')
    parser.add_argument('--no-persistent', action='store_true', help='Disable persistent storage')

    args = parser.parse_args()

    system = MultiAgentSystem(
        force_rebuild=args.force_rebuild,
        use_persistent=not args.no_persistent
    )

    try:
        if args.command == 'init':
            await system.initialize()
            print("System initialized successfully")

        elif args.command == 'query':
            if not args.query:
                print("Error: --query is required for query command")
                return

            await system.initialize()
            response = await system.process_query(
                args.query,
                args.user_id,
                not args.no_evaluation
            )

            print("\n" + "="*80)
            print(f"Query: {response['query']}")
            print(f"Department: {response['department']}")
            print(f"Agent: {response['agent_used']}")
            print(f"Confidence: {response['confidence']:.3f}")
            print(f"Processing Time: {response['processing_time']:.3f}s")
            print("\nAnswer:")
            print(response['answer'])

            if 'quality_evaluation' in response:
                qe = response['quality_evaluation']
                print(f"\nQuality Score: {qe['overall_score']}/10")
                print("Dimension Scores:")
                for dim, score in qe['dimension_scores'].items():
                    print(f"  {dim}: {score}/10")

            print("="*80)

        elif args.command == 'test':
            await system.initialize()
            results = await system.test_with_queries(
                args.file,
                not args.no_evaluation
            )

            if 'error' in results:
                print(f"Error: {results['error']}")
                return

            summary = results['test_summary']
            print("\n" + "="*80)
            print("TEST RESULTS SUMMARY")
            print("="*80)
            print(f"Total Queries: {summary['total_queries']}")
            print(f"Classification Accuracy: {summary['classification_accuracy']:.1%}")
            print(f"Average Confidence: {summary['avg_confidence']:.3f}")
            print(f"Average Processing Time: {summary['avg_processing_time']:.3f}s")
            print(f"Total Time: {summary['total_time']:.3f}s")
            if summary['avg_quality_score']:
                print(f"Average Quality Score: {summary['avg_quality_score']}/10")
            print("="*80)

        elif args.command == 'status':
            status = system.get_system_status()
            print("\n" + "="*80)
            print("SYSTEM STATUS")
            print("="*80)
            print(f"Overall Status: {status['status']}")
            if 'components' in status:
                for component, initialized in status['components'].items():
                    print(f"{component.capitalize()}: {'✅' if initialized else '❌'}")
            if 'available_departments' in status:
                print(f"Available Departments: {', '.join(status['available_departments'])}")
            print("="*80)

        elif args.command == 'cache-clear':
            system.clear_cache(args.department)

        elif args.command == 'cache-info':
            cache_info = system.get_cache_info()
            print("\n" + "="*80)
            print("CACHE INFORMATION")
            print("="*80)
            print(f"Cache Directory: {cache_info['cache_dir']}")
            print(f"Cache Exists: {'✅' if cache_info['exists'] else '❌'}")

            if cache_info['departments']:
                print(f"\nDepartment Caches:")
                for dept, info in cache_info['departments'].items():
                    print(f"\n  {dept.upper()}:")
                    print(f"    Embeddings: {'✅' if info['embeddings_cached'] else '❌'}")
                    print(f"    Index: {'✅' if info['index_cached'] else '❌'}")
                    print(f"    Metadata: {'✅' if info['metadata_cached'] else '❌'}")
                    print(f"    Size: {info['total_size_mb']:.2f} MB")
            else:
                print("\nNo cached data found.")
            print("="*80)

        elif args.command == 'store-clear':
            system.clear_store(args.department)

        elif args.command == 'store-info':
            store_info = system.get_store_info()
            print("\n" + "="*80)
            print("PERSISTENT STORE INFORMATION")
            print("="*80)
            print(f"Store Enabled: {'✅' if store_info['store_enabled'] else '❌'}")
            print(f"FAISS Indices: {store_info['faiss_indices_dir']}")
            print(f"Embeddings: {store_info['embeddings_dir']}")
            print(f"Metadata: {store_info['metadata_dir']}")

            if store_info['departments']:
                print(f"\nDepartment Stores:")
                for dept, info in store_info['departments'].items():
                    print(f"\n  {dept.upper()}:")
                    print(f"    Complete: {'✅' if info['complete'] else '❌'}")
                    print(f"    FAISS Index: {'✅' if info['faiss_index'] else '❌'}")
                    print(f"    Embeddings: {'✅' if info['embeddings'] else '❌'}")
                    print(f"    Metadata: {'✅' if info['metadata'] else '❌'}")
                    print(f"    Documents: {'✅' if info['documents'] else '❌'}")
                    print(f"    Size: {info['total_size_mb']:.2f} MB")
            else:
                print("\nNo persistent stores found.")
            print("="*80)

    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await system.shutdown()


if __name__ == "__main__":
    asyncio.run(main())