import traceback
import validators
import streamlit as st

from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import (
    YoutubeLoader,
    UnstructuredURLLoader,
)

# Streamlit App
st.set_page_config(
    page_title="LangChain: Summarize Text From YT or Website",
    page_icon="🦜"
)

st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader("Summarize URL")

# Sidebar
with st.sidebar:
    groq_api_key = st.text_input(
        "Groq API Key",
        type="password"
    )

# URL Input
generic_url = st.text_input(
    "URL",
    label_visibility="collapsed"
)

# LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=groq_api_key
)

# Prompt
prompt_template = """
Provide a summary of the following content in 300 words.

Content:
{text}
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["text"]
)

# Button
if st.button("Summarize the Content from YT or Website"):

    if not groq_api_key.strip():
        st.error("Please enter your Groq API Key.")

    elif not generic_url.strip():
        st.error("Please enter a URL.")

    elif not validators.url(generic_url):
        st.error("Please enter a valid URL.")

    else:
        try:
            with st.spinner("Loading..."):

                # Load documents
                if "youtube.com" in generic_url or "youtu.be" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(
                        generic_url,
                        add_video_info=True
                    )
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={
                            "User-Agent": "Mozilla/5.0"
                        },
                    )

                docs = loader.load()

                # Create summarize chain
                chain = load_summarize_chain(
                    llm,
                    chain_type="stuff",
                    prompt=prompt
                )

                # Run chain
                output = chain.invoke(
                    {"input_documents": docs}
                )

                st.success(output["output_text"])

        except Exception as e:
            st.error(str(e))
            st.code(traceback.format_exc())
