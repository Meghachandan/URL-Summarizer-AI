import re
import traceback
import streamlit as st
import validators

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_community.document_loaders import UnstructuredURLLoader
from youtube_transcript_api import YouTubeTranscriptApi

# -------------------- STREAMLIT --------------------

st.set_page_config(
    page_title="AI URL & YouTube Summarizer",
    page_icon="📝",
    layout="wide"
)

st.title("📝 AI URL & YouTube Summarizer")

with st.sidebar:
    groq_api_key = st.text_input(
        "Groq API Key",
        type="password"
    )

url = st.text_input("Enter Website or YouTube URL")

# -------------------- LLM --------------------

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=groq_api_key
)

# -------------------- FUNCTIONS --------------------

def extract_video_id(url):
    patterns = [
        r"v=([^&]+)",
        r"youtu\.be/([^?]+)",
        r"embed/([^?]+)"
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    return None


def load_youtube(video_url):
    video_id = extract_video_id(video_url)

    if not video_id:
        raise Exception("Invalid YouTube URL")

    api = YouTubeTranscriptApi()

    transcript_list = api.list(video_id)

    # Automatically pick the first available transcript
    transcript = next(iter(transcript_list))

    transcript_data = transcript.fetch()

    text = " ".join(
        snippet.text
        for snippet in transcript_data
    )

    return text


def load_website(site):
    loader = UnstructuredURLLoader(
        urls=[site],
        ssl_verify=False,
        headers={"User-Agent": "Mozilla/5.0"},
    )

    docs = loader.load()

    return "\n".join(doc.page_content for doc in docs)

# -------------------- BUTTON --------------------

if st.button("Summarize"):

    if not groq_api_key:
        st.error("Enter your Groq API Key.")
        st.stop()

    if not validators.url(url):
        st.error("Enter a valid URL.")
        st.stop()

    try:

        with st.spinner("Loading content..."):

            if "youtube.com" in url or "youtu.be" in url:
                text = load_youtube(url)
            else:
                text = load_website(url)

            prompt = f"""
You are an expert summarizer.

The content below may be in Telugu, Hindi,
English, or any other language.

If it is not in English,
first understand/translate it internally,
then provide the summary ONLY in English.

Provide:

• A concise summary
• Important points
• Final conclusion

Content:

{text}
"""

            response = llm.invoke(prompt)

            st.success("Summary Generated")

            st.write(response.content)

    except Exception as e:

        st.error(str(e))
        st.code(traceback.format_exc())
