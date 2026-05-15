from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb


# Create text splitter object
# This breaks large text into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,       # max characters per chunk
    chunk_overlap=50      # overlap helps preserve context
)

# Load embedding model
# This model converts text into vectors (embeddings)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Create ChromaDB client
# This acts as our vector database
chroma_client = chromadb.Client()

# Create collection inside vector DB
collection = chroma_client.get_or_create_collection(name="pdf_documents")


def process_text(text):

    # Split large text into chunks
    chunks = text_splitter.split_text(text)

    # Generate embeddings for each chunk
    embeddings = embedding_model.encode(chunks)

    # Store chunks + embeddings in vector database
    for index, (chunk, embedding) in enumerate(zip(chunks, embeddings)):

        collection.add(
            ids=[str(index)],
            documents=[chunk],
            embeddings=[embedding.tolist()]
        )

    return chunks