import re
import traceback
import streamlit as st
import validators

from langchain_groq import ChatGroq
from langchain_community.document_loaders import UnstructuredURLLoader
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    NoTranscriptFound,
    TranscriptsDisabled,
)

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
    groq_api_key=groq_api_key,
    temperature=0
)

# -------------------- FUNCTIONS --------------------

def extract_video_id(url):
    patterns = [
        r"v=([^&]+)",
        r"youtu\.be/([^?&]+)",
        r"embed/([^?&]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    return None


def load_youtube(video_url):
    video_id = extract_video_id(video_url)

    if not video_id:
        raise Exception("Invalid YouTube URL.")

    api = YouTubeTranscriptApi()

    try:
        # Latest youtube-transcript-api (v1.x)
        fetched = api.fetch(video_id)

        # Convert transcript to text
        try:
            text = " ".join(
                snippet.text for snippet in fetched
            )
        except Exception:
            text = " ".join(
                item["text"]
                for item in fetched.to_raw_data()
            )

        return text

    except NoTranscriptFound:
        raise Exception(
            "No transcript is available for this YouTube video."
        )

    except TranscriptsDisabled:
        raise Exception(
            "Transcripts are disabled for this video."
        )


def load_website(site):
    loader = UnstructuredURLLoader(
        urls=[site],
        ssl_verify=False,
        headers={
            "User-Agent": "Mozilla/5.0"
        },
    )

    docs = loader.load()

    text = "\n".join(doc.page_content for doc in docs)

    return text


# -------------------- BUTTON --------------------

if st.button("Summarize"):

    if not groq_api_key:
        st.error("Please enter your Groq API Key.")
        st.stop()

    if not validators.url(url):
        st.error("Please enter a valid URL.")
        st.stop()

    try:

        with st.spinner("Loading..."):

            if "youtube.com" in url or "youtu.be" in url:
                text = load_youtube(url)
            else:
                text = load_website(url)

            prompt = f"""
You are an expert AI summarizer.

The content below may be written in Telugu, Hindi,
English, or another language.

If it is not English, first understand or translate
it internally, then provide the final answer ONLY in English.

Generate:

1. Executive Summary
2. Important Key Points
3. Final Conclusion

Content:

{text}
"""

            response = llm.invoke(prompt)

        st.success("Summary Generated Successfully!")

        st.markdown(response.content)

    except Exception as e:
        st.error(str(e))
        st.code(traceback.format_exc())
