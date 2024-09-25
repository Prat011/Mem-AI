# Mem-AI

Mem - AI is a Python library for managing memory and context in Large Language Model applications. It provides efficient storage, retrieval, and semantic search capabilities to enhance the context-awareness of LLM interactions.

## Features

- Short-term and long-term memory management
- Efficient semantic search using FAISS
- Flexible embedding options with sentence-transformers
- Easy integration with various LLM platforms

## Installation

```bash
pip install llm-memory-manager
```

## Quick Start

```python
from llm_memory.core import LLMMemoryManager
from llm_memory.embedding.sentence_transformer import SentenceTransformerEmbedding
from llm_memory.storage.in_memory import InMemoryStorage

# Initialize components
embedding = SentenceTransformerEmbedding()
storage = InMemoryStorage()
embedding_dim = 384

# Create LLMMemoryManager instance
memory_manager = LLMMemoryManager(embedding, storage, embedding_dim)

# Add a memory
memory_manager.add_to_memory("The Eiffel Tower is in Paris, France.")

# Retrieve relevant memories
query = "What are some landmarks in Paris?"
results = memory_manager.get_relevant_memories(query, k=1)
print(results)
```

For more detailed usage examples, see the `examples/` directory.