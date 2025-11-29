#!/usr/bin/env python3
"""Test to check if the flush method exists."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.langfuse_client import LangfuseClient

# Create a new instance
client = LangfuseClient()

print("Methods in LangfuseClient:")
methods = [method for method in dir(client) if not method.startswith('_')]
for method in methods:
    print(f"  - {method}")

print(f"\nHas flush method: {hasattr(client, 'flush')}")

if hasattr(client, 'flush'):
    print("✅ Flush method found")
    try:
        client.flush()
        print("✅ Flush method executed successfully")
    except Exception as e:
        print(f"❌ Error executing flush: {e}")
else:
    print("❌ Flush method not found")