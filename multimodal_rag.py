import os
import uuid
import base64
from typing import List, Union, Optional
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer

class MultimodalRAG:
    def __init__(self, collection_name: str = "multimodal_collection", use_memory: bool = True, path: Optional[str] = None, url: Optional[str] = None):
        """
        Initialize the Multimodal RAG system.
        """
        if url:
             print(f"--- Initializing Qdrant at {url} ---")
             self.client = QdrantClient(url=url)
        elif use_memory:
            print("--- Initializing Qdrant in memory ---")
            self.client = QdrantClient(":memory:")
        else:
            path = path or "./qdrant_db"
            print(f"--- Initializing Qdrant at {path} ---")
            self.client = QdrantClient(path=path)

        self.collection_name = collection_name
        
        # Load CLIP model
        print("--- Loading CLIP model (clip-ViT-B-32) ---")
        self.model = SentenceTransformer('clip-ViT-B-32')
        self.vector_size = 512  # clip-ViT-B-32 dimension

        # Create collection
        self._create_collection_if_not_exists()

    def _create_collection_if_not_exists(self):
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        
        if not exists:
            print(f"--- Creating collection '{self.collection_name}' ---")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE
                )
            )
        else:
            print(f"--- Collection '{self.collection_name}' already exists ---")

    def index_images(self, image_paths: List[str], metadatas: Optional[List[dict]] = None):
        """
        Embed and index images.
        """
        print(f"--- Indexing {len(image_paths)} images ---")
        images = [Image.open(path) for path in image_paths]
        embeddings = self.model.encode(images)
        
        points = []
        for i, (path, emb) in enumerate(zip(image_paths, embeddings)):
            payload = {"type": "image", "path": path}
            if metadatas and i < len(metadatas):
                payload.update(metadatas[i])
                
            points.append(models.PointStruct(
                id=str(uuid.uuid4()),
                vector=emb.tolist(),
                payload=payload
            ))
            
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        print("--- Indexing complete ---")

    def index_texts(self, texts: List[str], metadatas: Optional[List[dict]] = None):
        """
        Embed and index texts.
        """
        print(f"--- Indexing {len(texts)} texts ---")
        embeddings = self.model.encode(texts)
        
        points = []
        for i, (text, emb) in enumerate(zip(texts, embeddings)):
            payload = {"type": "text", "content": text}
            if metadatas and i < len(metadatas):
                payload.update(metadatas[i])
                
            points.append(models.PointStruct(
                id=str(uuid.uuid4()),
                vector=emb.tolist(),
                payload=payload
            ))
            
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        print("--- Indexing complete ---")

    def search(self, query: Union[str, Image.Image], k: int = 3, score_threshold: float = 0.0):
        """
        Search for similar items using text or image query.
        """
        print(f"--- Searching... ---")
        if isinstance(query, str):
            query_vector = self.model.encode(query)
        elif isinstance(query, Image.Image):
            query_vector = self.model.encode(query)
        else:
            try:
                # Try opening as path
                img = Image.open(query)
                query_vector = self.model.encode(img)
            except Exception:
                raise ValueError("Query must be string, PIL Image, or path to image.")

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector.tolist(),
            limit=k,
            with_payload=True,
            score_threshold=score_threshold
        ).points
        
        return results

    def _encode_image(self, image_path: str):
        # Open image using PIL
        with Image.open(image_path) as img:
            # Convert to RGB if needed (e.g. for PNG with transparency or AVIF)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Save to a BytesIO object as JPEG
            from io import BytesIO
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def generate_answer(self, query: str, retrieved_results: list):
        """
        Generates an answer using GPT-4o based on certain retrieved images.
        """
        print("--- Generating Answer with GPT-4o ---")
        
        if not retrieved_results:
            return "No information found to answer the question."
            
        from openai import OpenAI
        from dotenv import load_dotenv
        
        load_dotenv() # Load .env file if present
        client = OpenAI() # Expects OPENAI_API_KEY in env
        
        # Prepare messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Answer this question: {query}. If the answer is not in the images, say 'I don't know'"}
                ]
            }
        ]
        
        # Add images to context
        # We limit to top 2 to avoid hitting token limits
        for res in retrieved_results[:2]:
            path = res.payload.get('path')
            if path and os.path.exists(path):
                base64_image = self._encode_image(path)
                messages[0]["content"].append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                })
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=300
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating answer: {str(e)}"

    def clear_collection(self):
        """Deletes and recreates the collection"""
        self.client.delete_collection(self.collection_name)
        self._create_collection_if_not_exists()
