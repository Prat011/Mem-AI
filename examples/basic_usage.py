from llm_memory.core import LLMMemoryManager
from llm_memory.embedding.sentence_transformer import SentenceTransformerEmbedding
from llm_memory.storage.in_memory import InMemoryStorage

# Initialize components
embedding = SentenceTransformerEmbedding()
storage = InMemoryStorage()
embedding_dim = 384  # Dimension of the chosen embedding model

# Create LLMMemoryManager instance
memory_manager = LLMMemoryManager(embedding, storage, embedding_dim)

# Add some memories
memory_manager.add_to_memory("The Eiffel Tower is in Paris, France.")
memory_manager.add_to_memory("The Louvre Museum houses the Mona Lisa.")

# Perform a search
query = "What are some landmarks in Paris?"
results = memory_manager.get_relevant_memories(query, k=2, threshold=0.5, return_scores=True)

for text, score in results:
    print(f"Text: {text}, Similarity: {score:.2f}")