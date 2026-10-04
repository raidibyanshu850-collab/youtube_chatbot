import re
import streamlit as st

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser


st.set_page_config(page_title="YouTube RAG Assistant", page_icon="▶️", layout="centered")
st.title("▶️ YouTube RAG Assistant")
st.caption("Process a video's transcript, then ask questions grounded in its content.")

def extract_video_id(url: str):
    """Extract a YouTube video ID from common YouTube URL formats."""
    patterns = [
        r"(?:youtube\.com/watch\?.*?v=)([A-Za-z0-9_-]{11})",
        r"(?:youtu\.be/)([A-Za-z0-9_-]{11})",
        r"(?:youtube\.com/embed/)([A-Za-z0-9_-]{11})",
        r"(?:youtube\.com/shorts/)([A-Za-z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_transcript(video_id: str):
    # Current youtube-transcript-api interface used in the supplied notebook
    api = YouTubeTranscriptApi()
    transcript = api.fetch(video_id, languages=["en"])
    return " ".join(snippet.text for snippet in transcript.snippets)

def build_chain(transcript_text: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(transcript_text)
    if not chunks:
        raise ValueError("The transcript was empty.")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_store = FAISS.from_texts(chunks, embeddings)
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    llm = OllamaLLM(model="llama3.2:latest")
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are a helpful assistant. Answer ONLY from the provided transcript context. "
            "If the context is insufficient, just say you don't know.\n\n"
            "Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        ),
    )
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    parallel_chain = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    })
    return parallel_chain | prompt | llm | StrOutputParser(), len(chunks)

if "chain" not in st.session_state:
    st.session_state.chain = None
if "video_id" not in st.session_state:
    st.session_state.video_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.container(border=True):
    st.subheader("1. Add a YouTube video")
    url = st.text_input("Paste YouTube video URL", placeholder="https://www.youtube.com/watch?v=...")
    video_id = extract_video_id(url.strip()) if url.strip() else None
    if url.strip():
        if video_id:
            st.success(f"Video ID detected: `{video_id}`")
        else:
            st.error("That doesn't look like a valid YouTube video URL.")
    process = st.button("Process video", type="primary", disabled=not bool(video_id), use_container_width=True)

if process and video_id:
    try:
        with st.spinner("Fetching transcript and building the search index..."):
            transcript_text = get_transcript(video_id)
            chain, chunk_count = build_chain(transcript_text)
        st.session_state.chain = chain
        st.session_state.video_id = video_id
        st.session_state.messages = []
        st.success(f"Video ready! Indexed {chunk_count} transcript chunks.")
    except TranscriptsDisabled:
        st.error("Captions are disabled for this video.")
    except NoTranscriptFound:
        st.error("No English transcript was found. This version currently supports English transcripts.")
    except Exception as e:
        st.error(f"Could not process the video: {e}")
        st.info("Check that Ollama is running and that `llama3.2:latest` and `nomic-embed-text` are available.")

if st.session_state.chain:
    st.divider()
    st.subheader("2. Ask questions")
    st.caption(f"Current video ID: {st.session_state.video_id}")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask something about this video...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Searching the transcript and generating an answer..."):
                try:
                    answer = st.session_state.chain.invoke(question)
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Error generating answer: {e}")
else:
    st.info("Process a video first. The chat will appear here once its transcript has been indexed.")

with st.expander("Setup / troubleshooting"):
    st.markdown(
        "- Start Ollama before launching the app.\n"
        "- Pull the models with `ollama pull llama3.2:latest` and `ollama pull nomic-embed-text`.\n"
        "- This app currently requests English transcripts. Videos without an accessible English transcript cannot be indexed.\n"
        "- Run with `streamlit run app.py`."
    )
