import re
import traceback
import validators
import streamlit as st

from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import UnstructuredURLLoader
from youtube_transcript_api import YouTubeTranscriptApi

# ---------------- Streamlit ----------------

st.set_page_config(
    page_title="AI URL Summarizer",
    page_icon="📝",
    layout="wide"
)

st.title("📝 AI URL & YouTube Summarizer")

with st.sidebar:
    groq_api_key = st.text_input(
        "Groq API Key",
        type="password"
    )

generic_url = st.text_input(
    "Enter Website or YouTube URL"
)

# ---------------- LLM ----------------

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=groq_api_key
)

# ---------------- Prompt ----------------

prompt = PromptTemplate(
    template="""
Summarize the following content in about 300 words.

Content:
{text}
""",
    input_variables=["text"]
)

# ---------------- Helper ----------------

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


# ---------------- Button ----------------

if st.button("Summarize"):

    if not groq_api_key:
        st.error("Enter your Groq API Key.")

    elif not generic_url:
        st.error("Enter a URL.")

    elif not validators.url(generic_url):
        st.error("Enter a valid URL.")

    else:

        try:

            with st.spinner("Loading content..."):

                # ---------------- YouTube ----------------

                if (
                    "youtube.com" in generic_url
                    or "youtu.be" in generic_url
                ):

                    video_id = extract_video_id(generic_url)

                    if video_id is None:
                        st.error("Invalid YouTube URL.")
                        st.stop()

                    api = YouTubeTranscriptApi()

                    transcript = api.fetch(
                        video_id,
                        languages=["en"]
                    )

                    text = " ".join(
                        snippet.text
                        for snippet in transcript
                    )

                    docs = [
                        Document(page_content=text)
                    ]

                # ---------------- Website ----------------

                else:

                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={
                            "User-Agent": "Mozilla/5.0"
                        }
                    )

                    docs = loader.load()

                # ---------------- Chain ----------------

                chain = load_summarize_chain(
                    llm=llm,
                    chain_type="stuff",
                    prompt=prompt
                )

                result = chain.invoke(
                    {
                        "input_documents": docs
                    }
                )

                st.success("Summary Generated")

                st.write(result["output_text"])

        except Exception as e:

            st.error(str(e))
            st.code(traceback.format_exc())
