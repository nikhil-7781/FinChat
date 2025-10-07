import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from sentence_transformers import SentenceTransformer
import pinecone
from dotenv import load_dotenv
import os
from pinecone import ServerlessSpec

load_dotenv()

class PDFLoader:
    model=SentenceTransformer("all-MiniLM-L6-v2")
    def __init__(self, pdf_path):
        self.pdf_path=pdf_path
    
    def extract_text(self):
        doc=fitz.open(self.pdf_path)
        text=""
        for page in doc:
            text+=page.get_text()
        return text

    def create_embeddings(self, text):
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs=text_splitter.split_text(text)
        embeddings = self.model.encode(docs, batch_size=32, convert_to_numpy=True, show_progress_bar=True)
        return docs, embeddings

class PineconeStore:
    def __init__(self):
        pinecone_api=os.getenv('PINECONE_API')
        self.pc=pinecone.Pinecone(api_key=pinecone_api)
        self.index_name="pdf-1-vec-store-v2"
        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=384,
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )
    def save_vectors(self, vectors, metadata, docs, batch_size=50):
        index=self.pc.Index(self.index_name)
        upserts=[]

        for i, vector in enumerate(vectors):
            vector_id = f"{metadata['id']}_chunk_{i}"  # Unique ID for each chunk
            chunk_metadata = {
                "id": vector_id,
                "source": metadata["source"],
                "chunk": i,
                "text": docs[i][:500] # Add the text of the chunk here
            }
            upserts.append((vector_id, vector.tolist(), chunk_metadata))
            if len(upserts) >= batch_size:
                index.upsert(vectors=upserts)
                upserts = []
        # send remaining
        if upserts:
            index.upsert(vectors=upserts)
    
if __name__=="__main__":
    loader=PDFLoader("static_knowledge/The Intelligent Investor - BENJAMIN GRAHAM.pdf")
    text=loader.extract_text()
    docs, embeddings=loader.create_embeddings(text)
    vector_store=PineconeStore()
    vector_store.save_vectors(embeddings, {"id": "doc_1", "source": "example.pdf"}, docs)

    