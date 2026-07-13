import re
import traceback

import streamlit as st
import validators

from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import UnstructuredURLLoader
from youtube_transcript_api import YouTubeTranscriptApi

# -------------------------
# Streamlit Config
# -------------------------
st.set_page_config(
    page_title="AI URL & YouTube Summarizer",
    page_icon="📝",
    layout="wide"
)

st.title("AI URL & YouTube Summarizer")
st.write("Summarize any YouTube video or Website using Groq LLM.")

# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    groq_api_key = st.text_input(
        "Groq API Key",
        type="password"
    )

# -------------------------
# URL Input
# -------------------------
generic_url = st.text_input(
    "Enter Website or YouTube URL"
)

# -------------------------
# LLM
# -------------------------
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=groq_api_key
)

# -------------------------
# Prompt
# -------------------------
prompt_template = """
You are an expert summarizer.

Summarize the following content in approximately 300 words.

Content:
{text}
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["text"]
)

# -------------------------
# Helper Function
# -------------------------
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

# -------------------------
# Button
# -------------------------
if st.button("Summarize"):

    if not groq_api_key:
        st.error("Please enter your Groq API Key.")

    elif not generic_url:
        st.error("Please enter a URL.")

    elif not validators.url(generic_url):
        st.error("Please enter a valid URL.")

    else:

        try:

            with st.spinner("Loading content..."):

                # -------------------------
                # YouTube
                # -------------------------
                if (
                    "youtube.com" in generic_url
                    or "youtu.be" in generic_url
                ):

                    video_id = extract_video_id(generic_url)

                    if video_id is None:
                        st.error("Invalid YouTube URL.")
                        st.stop()

                    transcript = YouTubeTranscriptApi.get_transcript(video_id)

                    text = " ".join(
                        [item["text"] for item in transcript]
                    )

                    docs = [
                        Document(page_content=text)
                    ]

                # -------------------------
                # Website
                # -------------------------
                else:

                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={
                            "User-Agent": "Mozilla/5.0"
                        },
                    )

                    docs = loader.load()

                # -------------------------
                # Summarization Chain
                # -------------------------
                chain = load_summarize_chain(
                    llm=llm,
                    chain_type="stuff",
                    prompt=prompt,
                )

                result = chain.invoke(
                    {"input_documents": docs}
                )

                st.success("Summary Generated Successfully!")

                st.write(result["output_text"])

        except Exception as e:

            st.error(str(e))
            st.code(traceback.format_exc())
