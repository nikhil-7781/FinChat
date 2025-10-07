from docparse import create_embeddings
import pinecone
from dotenv import load_dotenv
import os
from pinecone import ServerlessSpec

load_dotenv()

class PineconeStore:
    def __init__(self):
        pinecone_api=os.getenv('PINECONE_API')
        self.pc=pinecone(api_key=pinecone_api)
        self.index_name="pdf-1-vec-store"
        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=1536,
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )
    def save_vectors(self, vectors, metadata, docs):
        index=self.pc.Index(self.index_name)

        for i, vector in enumerate(vectors):
            vector_id = f"{metadata['id']}_chunk_{i}"  # Unique ID for each chunk
            chunk_metadata = {
                "id": vector_id,
                "source": metadata["source"],
                "chunk": i,
                "text": docs[i]  # Add the text of the chunk here
            }
            # Upsert each vector with its corresponding metadata
            index.upsert(vectors=[(vector_id, vector, chunk_metadata)])

if __name__=="__main__":
    vector_store=PineconeStore()
    docs, embedding=create_embeddings()
    vector_store.save_vectors(embedding, {"id": "doc_1", "source": "example.pdf"}, docs)

