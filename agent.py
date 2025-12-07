import os
import time
import requests
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

# ==========================================
# 🛡️ USER-AGENT PATCH (Place this at the top)
# ==========================================
# Define a real browser user agent to fool YouTube
BROWSER_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"


def get_patched_session():
    """Factory to create a requests session that looks like Chrome."""
    session = requests.Session()
    session.headers.update({"User-Agent": BROWSER_USER_AGENT})
    return session


# OVERRIDE the requests library globally
# This effectively 'tricks' youtube_transcript_api into using our headers
requests.Session = get_patched_session
# =======


# Load env variables
load_dotenv()

# Ensure API Key is present
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY is missing from environment variables")


class YotubeRAG:
    def __init__(self):
        # Using a slightly more capable model for summary if available, or stick to 8b
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.3,  # Lower temperature for more factual summaries
            max_retries=2
        )

        # --- PROMPTS ---
        self.summary_prompt = PromptTemplate(
            template="""
            Summarize the following video transcript chunk concisely. Capture key points, main arguments, and conclusions.
            
            Transcript Chunk:
            {content}

            Important Note:
            Reply Only in English Inrrespective of input language 
            
            Summary:""",
            input_variables=["content"]
        )

        self.final_summary_prompt = PromptTemplate(
            template="""
            Here are summaries of different parts of a video. Combine them into one coherent, detailed final summary.
            
            Partial Summaries:
            {content}

            Important Note:
            Reply Only in English Inrrespective of input language 
            
            Final Detailed Summary:""",
            input_variables=["content"]
        )

        self.qa_prompt = PromptTemplate(
            template="""
            You are an AI assistant answering questions about a video. Use the provided context ONLY.
            
            Context:
            {context}
            
            Question:
            {question}
            
            If the answer is not in the context, say "I don't know based on the video context."
            Keep answers concise and professional.

            Important Note:
            Reply Only in English Inrrespective of input language 
            """,
            input_variables=["context", "question"]
        )

    def fetch_youtube_transcript(self, video_id):
        try:
            transcript_list = YouTubeTranscriptApi().fetch(
                video_id=video_id, languages=["en", "hi"]).to_raw_data()
            transcript = " ".join([t['text'] for t in transcript_list])
            return transcript
        except (TranscriptsDisabled, NoTranscriptFound):
            raise Exception(
                "Subtitles are disabled or not available for this video.")
        except Exception as e:
            print('error ', e)
            raise Exception(f"YouTube Error: {str(e)}")

    def splitText(self, transcript: str, chunk_size=4000, chunk_overlap=200):
        # Optimized for standard RAG retrieval
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        return splitter.create_documents([transcript])

    def create_retreiver(self, texts):
        # Using a lightweight, fast embedding model
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        vector_store = FAISS.from_documents(texts, embedding=embeddings)
        return vector_store.as_retriever(search_kwargs={"k": 3})

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def generate_smart_summary(self, transcript):
        """
        Handles Token Limits by chunking long videos automatically.
        """
        # 1. Estimate Token Count (Approx 1 token = 4 chars)
        # Groq TPM Limit is tight (6000). We need to be careful.
        # Safe chunk size ~ 12,000 chars (~3000 tokens) to leave room for output and prompt.
        MAX_CHARS_PER_CHUNK = 10000

        if len(transcript) < MAX_CHARS_PER_CHUNK:
            # Short video: Direct summary
            chain = self.summary_prompt | self.llm | StrOutputParser()
            return chain.invoke({"content": transcript})

        # Long video: Map-Reduce Strategy
        print(
            f"Video too long ({len(transcript)} chars). Switching to Chunked Summary...")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=MAX_CHARS_PER_CHUNK, chunk_overlap=500)
        docs = splitter.create_documents([transcript])

        partial_summaries = []
        map_chain = self.summary_prompt | self.llm | StrOutputParser()

        for i, doc in enumerate(docs):
            print(f"Summarizing chunk {i+1}/{len(docs)}...")
            try:
                summary = map_chain.invoke({"content": doc.page_content})
                partial_summaries.append(summary)
                # CRITICAL: Sleep to respect Rate Limits (TPM)
                time.sleep(2)
            except Exception as e:
                print(f"Error summarzing chunk {i}: {e}")
                continue

        # Combine summaries
        combined_text = "\n\n".join(partial_summaries)
        reduce_chain = self.final_summary_prompt | self.llm | StrOutputParser()
        return reduce_chain.invoke({"content": combined_text})

    def create_qa_chain(self, retriever):
        return (
            {"context": retriever | self.format_docs,
                "question": RunnablePassthrough()}
            | self.qa_prompt
            | self.llm
            | StrOutputParser()
        )
