import os
from PIL import Image, ImageDraw
from multimodal_rag import MultimodalRAG
from dotenv import load_dotenv
load_dotenv()

def get_images_from_dir(image_dir="data/images"):
    if not os.path.exists(image_dir):
        print(f"Directory {image_dir} does not exist.")
        return []
    
    valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".avif", ".bmp"}
    image_paths = []
    for f in os.listdir(image_dir):
        ext = os.path.splitext(f)[1].lower()
        if ext in valid_extensions:
            image_paths.append(os.path.join(image_dir, f))
            
    return image_paths

def main():
    print("=== Multimodal RAG Test with User Data ===")
    
    # 1. Init RAG - Connecting to Docker instance
    try:
        rag = MultimodalRAG(url="http://localhost:6333", use_memory=False)
        # Clear collection to start fresh for testing
        rag.clear_collection()
    except Exception as e:
        print(f"Failed to connect to Docker Qdrant: {e}")
        print("Falling back to memory...")
        rag = MultimodalRAG(use_memory=True)
    
    # 2. Get user images
    image_paths = get_images_from_dir()
    if not image_paths:
        print("No images found in data/images. Please add some images.")
        return

    processed_paths = [os.path.abspath(p) for p in image_paths]
    print(f"Found {len(processed_paths)} images.")
    
    # 3. Index Images
    # We don't have metadata for these, so we'll just index them without specific metadata for now
    rag.index_images(processed_paths)
    
    # 4. Search with Text
    queries = ["Who Quoted you are never too old to set another goal or to dream a new dream"]
    
    for q in queries:
        print(f"\nQuery: '{q}'")
        results = rag.search(q, k=2)
        if results:
            for res in results:
                print(f"Match: {os.path.basename(res.payload['path'])} (Score: {res.score:.4f})")
            
            # Generate Answer
            answer = rag.generate_answer(q, results)
            print(f"\nGenerated Answer:\n{answer}")
        else:
            print("No results found.")

    # 5. Index Texts and search with Image (Cross-modal)
    print("\n--- Testing Text Indexing & Image Search ---")
    texts = ["A photo of a dog", "A photo of a cat", "A photo of a car"]
    rag.index_texts(texts)
    
    # Search with the red image (should effectively find nothing relevant ideally, or maybe just nearest neighbor)
    # But let's try searching for "dog" using text query first to verify text indexing
    results = rag.search("canine", k=1)
    print(f"\nQuery: 'canine'")
    print(f"Top Match: {results[0].payload['content']} (Score: {results[0].score:.4f})")

if __name__ == "__main__":
    main()
