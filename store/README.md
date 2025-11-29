# FAISS Index Store

This directory contains persistent FAISS indices for the multi-agent system.

## Directory Structure:
- `faiss_indices/` - FAISS index files (.index)
- `embeddings/` - Pre-computed embeddings (.npy)
- `metadata/` - Document metadata and chunk information (.json)

## Files by Department:
- `hr/` - Human Resources documents
- `tech/` - IT/Technology documents
- `finance/` - Finance documents

Each department directory contains:
- `faiss.index` - The FAISS vector index
- `embeddings.npy` - Document embeddings matrix
- `metadata.json` - Document metadata and chunk mappings
- `documents.json` - Original document content for reference

## Usage:
The system will automatically load existing indices if available.
To force rebuilding, delete the department directory or use the --force-rebuild flag.

