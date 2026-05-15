import os
from google import genai

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Import Chroma collection from rag_service
from services.rag_service import collection


# Load environment variables
load_dotenv()

# Configure Gemini API
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


# Load same embedding model used during storage
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def ask_question(question):

    # Convert user question into embedding vector
    question_embedding = embedding_model.encode(question)

    # Search vector DB for similar chunks
    results = collection.query(
        query_embeddings=[question_embedding.tolist()],
        n_results=3
    )

    # Extract retrieved chunks
    retrieved_chunks = results["documents"][0]

    # Combine chunks into one context block
    context = "\n\n".join(retrieved_chunks)

    # Prompt engineering
    # Force model to answer only from retrieved context
    prompt = f"""
    You are an AI assistant.

    Answer ONLY from the provided context.

    If answer is not present in context, say:
    "I could not find relevant information in the document."

    Context:
    {context}

    Question:
    {question}
    """

    try:
        # Generate response from Gemini
        response = client.models.generate_content(
         model="gemini-2.5-flash",
         contents=prompt
        )

        return {
          "answer": response.text,
          "retrieved_chunks": retrieved_chunks
        }
    except Exception as e:
        return {
            "error": str(e)
        }