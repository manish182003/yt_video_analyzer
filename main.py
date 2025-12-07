from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict
import re
import uuid
import time
import os
import shutil
import traceback
# Import necessary LangChain components for persistence
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from agent import YotubeRAG

app = FastAPI()

# --- CORS (Allow all for development, restrict for production) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- SESSION MANAGEMENT ---
# Store: { "session_id": { "chain": chain_obj, "timestamp": time.time() } }
sessions: Dict[str, dict] = {}
SESSION_DIR = "faiss_indexes"  # Directory to store persistent sessions

# Ensure session directory exists
os.makedirs(SESSION_DIR, exist_ok=True)


def cleanup_sessions():
    """Prevent memory leaks by removing old sessions from RAM and Disk"""
    # 1. Clear RAM if too full
    if len(sessions) > 100:
        print("Cleaning up RAM sessions...")
        sessions.clear()

    # 2. Clear Disk (Optional: delete files older than 24 hours)
    # This prevents the disk from filling up in production
    current_time = time.time()
    for session_id in os.listdir(SESSION_DIR):
        session_path = os.path.join(SESSION_DIR, session_id)
        if os.path.isdir(session_path):
            # If folder is older than 24 hours (86400 seconds), delete it
            if current_time - os.path.getmtime(session_path) > 86400:
                try:
                    shutil.rmtree(session_path)
                    print(f"Deleted old session from disk: {session_id}")
                except Exception as e:
                    print(f"Error cleaning disk: {e}")

# --- DATA MODELS ---


class VideoRequest(BaseModel):
    video_url: str = Field(..., description='YouTube Video URL')


class ChatRequest(BaseModel):
    session_id: str = Field(...,
                            description='Session ID from summarize endpoint')
    question: str = Field(..., description='User question')


@app.get("/")
def home():
    return {"status": "YouTube RAG Server Running"}


@app.post("/summarize")
def generate_summary(request: VideoRequest):
    cleanup_sessions()  # Run cleanup check

    try:
        # 1. Extract Video ID
        video_url = request.video_url
        match = re.search(r'v=([^&]+)', video_url)
        if match:
            video_id = match.group(1)
        else:
            # Handle "youtu.be/" format as well
            if "youtu.be/" in video_url:
                video_id = video_url.split("youtu.be/")[1].split("?")[0]
            else:
                raise HTTPException(
                    status_code=400, detail="Invalid YouTube URL")

        print(f"--- Processing {video_id} ---")

        # 2. Init Agent
        yt_agent = YotubeRAG()

        # 3. Fetch Transcript
        try:
            transcript = yt_agent.fetch_youtube_transcript(video_id)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

        # 4. Generate Summary (Smart Chunking handles the Rate Limit)
        try:
            summary = yt_agent.generate_smart_summary(transcript)
        except Exception as e:
            print(f"Summary Generation Error: {e}")
            raise HTTPException(
                status_code=500, detail="Failed to generate summary (Rate Limit or Model Error).")

        # 5. Create RAG Chain for Chat
        # We split text into smaller chunks for Vector Store retrieval
        texts = yt_agent.splitText(transcript)
        retriever = yt_agent.create_retreiver(texts)

        # CRITICAL FIX: Reduce k to 3 to prevent Rate Limit Errors (500)
        # Groq free tier has 6000 TPM limit. k=5 pushes it over the edge.
        retriever.search_kwargs['k'] = 3

        # --- PERSISTENCE LAYER ---
        # Generate Session ID
        session_id = str(uuid.uuid4())

        # Save the FAISS index to disk so it survives restarts
        try:
            save_path = os.path.join(SESSION_DIR, session_id)
            # Access the underlying vectorstore from the retriever and save it
            retriever.vectorstore.save_local(save_path)
        except Exception as e:
            print(f"Warning: Could not save to disk: {e}")

        qa_chain = yt_agent.create_qa_chain(retriever)

        # 6. Save Session to Memory
        sessions[session_id] = {
            "chain": qa_chain,
            "timestamp": time.time()
        }

        return {
            "summary": summary,
            "session_id": session_id,
            "message": "Summary generated successfully."
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Critical Error: {e}")
        traceback.print_exc()  # Print full error to console for debugging
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
def chat_with_video(request: ChatRequest):
    # 1. Try to find session in RAM
    if request.session_id in sessions:
        session_data = sessions[request.session_id]
        chain = session_data["chain"]
    else:
        # 2. If not in RAM, try to restore from Disk (Handling the Restart Issue)
        session_path = os.path.join(SESSION_DIR, request.session_id)
        if os.path.exists(session_path):
            print(f"Restoring session {request.session_id} from disk...")
            try:
                # Re-initialize the embeddings (Must match agent.py)
                embeddings = HuggingFaceEmbeddings(
                    model_name="BAAI/bge-small-en-v1.5")

                # Load the Vector Store from disk
                # allow_dangerous_deserialization is needed for some FAISS versions; safe here since we created it
                vectorstore = FAISS.load_local(
                    session_path, embeddings, allow_dangerous_deserialization=True)

                # Re-create the chain with REDUCED k=3
                retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
                yt_agent = YotubeRAG()
                chain = yt_agent.create_qa_chain(retriever)

                # Cache it back to RAM for next time
                sessions[request.session_id] = {
                    "chain": chain,
                    "timestamp": time.time()
                }
            except Exception as e:
                print(f"Failed to restore session: {e}")
                raise HTTPException(
                    status_code=500, detail="Could not restore session.")
        else:
            # 3. Truly not found
            raise HTTPException(
                status_code=404, detail="Session expired or not found. Please reload.")

    try:
        # Invoke Chain
        response = chain.invoke(request.question)

        return {
            "answer": response,
            "session_active": True
        }

    except Exception as e:
        print(f"Chat Error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"AI Error: {str(e)}")
