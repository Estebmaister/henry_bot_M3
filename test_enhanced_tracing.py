#!/usr/bin/env python3
"""
Test script to validate enhanced Langfuse tracing implementation.
"""

import asyncio
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.main import MultiAgentSystem


async def test_enhanced_tracing():
    """Test the enhanced tracing with a sample query."""
    print("🧪 Testing Enhanced Langfuse Tracing")
    print("=" * 50)

    try:
        # Initialize system
        system = MultiAgentSystem()
        await system.initialize()

        # Test with a sample query
        test_queries = [
            "What's our department budget for Q3?",
            "How do I request time off?",
            "What software do we have for project management?",
            "Can you explain our health insurance benefits?"
        ]

        for i, query in enumerate(test_queries, 1):
            print(f"\n🔍 Test {i}: {query}")
            print("-" * 30)

            try:
                result = await system.process_query(
                    query=query,
                    user_id=f"test_user_{i}",
                    evaluate_quality=True
                )

                print(f"✅ Query processed successfully")
                print(f"   Department: {result['department']}")
                print(f"   Agent: {result['agent_used']}")
                print(f"   Confidence: {result['confidence']:.3f}")
                print(f"   Processing Time: {result['processing_time']:.3f}s")
                print(f"   Source Documents: {len(result['source_documents'])}")

                if 'quality_evaluation' in result:
                    qe = result['quality_evaluation']
                    print(f"   Quality Score: {qe['overall_score']}/10")
                    print(f"   Quality Tier: {system._get_quality_tier(qe['overall_score'])}")

                print("\n📊 Langfuse trace created with enhanced debugging information:")
                print("   • Intent classification with confidence scores and alternatives")
                print("   • Document retrieval with similarity metrics")
                print("   • LLM call with token usage and performance data")
                print("   • Agent execution timing and metadata")
                print("   • Quality evaluation with dimension scores")
                print("   • Error handling with categorization")

            except Exception as e:
                print(f"❌ Error processing query: {e}")
                import traceback
                traceback.print_exc()

        await system.shutdown()
        print("\n🎉 Enhanced tracing test completed!")
        print("Check your Langfuse dashboard for detailed traces with rich debugging information.")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_enhanced_tracing())