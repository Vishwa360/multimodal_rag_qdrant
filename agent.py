import whisper
import jiwer
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional
import os
import warnings

# Suppress warnings for cleaner output (Whisper can be verbose about FP16 on CPU)
warnings.filterwarnings("ignore")

# Define the State
class AgentState(TypedDict):
    audio_path: str
    reference_text: Optional[str]
    transcription: Optional[str]
    wer_score: Optional[float]
    model_size: str

# Define the Nodes
def load_and_transcribe(state: AgentState):
    """
    Loads the Whisper model and transcribes the audio file found at audio_path.
    """
    audio_path = state["audio_path"]
    model_size = state.get("model_size", "base")
    
    print(f"--- Loading Whisper model ({model_size}) ---")
    model = whisper.load_model(model_size)
    
    print(f"--- Transcribing {audio_path} ---")
    if not os.path.exists(audio_path):
        return {"transcription": "Error: File not found."}
    
    result = model.transcribe(audio_path)
    text = result["text"].strip()
    
    return {"transcription": text}

def evaluate_transcription(state: AgentState):
    """
    Calculates the Word Error Rate (WER) if reference text is provided.
    """
    transcription = state.get("transcription", "")
    reference = state.get("reference_text", "")
    
    if not reference:
        print("--- No reference text provided, skipping evaluation ---")
        return {"wer_score": None}
        
    print(f"--- Evaluating Transcription ---")
    print(f"Reference: {reference}")
    print(f"Hypothesis: {transcription}")
    
    # Simple normalization to improve fairness of comparison
    # jiwer 4.0 requires the pipeline to end with ReduceToListOfListOfWords
    transformation = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.ReduceToListOfListOfWords(),
    ])
    
    wer = jiwer.wer(
        reference, 
        transcription, 
        reference_transform=transformation, 
        hypothesis_transform=transformation
    )
    
    print(f"WER Score: {wer:.4f}")
    return {"wer_score": wer}

# Build the Graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("transcriber", load_and_transcribe)
workflow.add_node("evaluator", evaluate_transcription)

# Set entry point
workflow.set_entry_point("transcriber")

# Connect nodes
workflow.add_edge("transcriber", "evaluator")
workflow.add_edge("evaluator", END)

# Compile
app = workflow.compile()

if __name__ == "__main__":
    # Example usage
    # We use the text from generate_data.py as the reference
    reference_text = "Hello, this is a test. We are checking if the speech to text agent works via LangGraph."
    
    input_data = {
        "audio_path": "data/sample.mp3",
        "model_size": "base",
        "reference_text": reference_text
    }
    
    print("Starting Agent...")
    result = app.invoke(input_data)
    print("\n--- Final Result ---")
    print(f"Transcription: {result['transcription']}")
    if result['wer_score'] is not None:
        print(f"WER: {result['wer_score']:.4f}")
