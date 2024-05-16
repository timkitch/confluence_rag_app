import streamlit as st
from dotenv import load_dotenv

# Import the ConfluenceQA class
from confluence_qa import ConfluenceQA


load_dotenv()

st.set_page_config(
    page_title='Q&A Bot for Confluence Page',
    page_icon='⚡',
    layout='wide',
    initial_sidebar_state='auto',
)
if "config" not in st.session_state:
    st.session_state["config"] = {}

# Initialize chat history if it doesn't exist
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = ""

if "confluence_qa" not in st.session_state:
    confluence_qa = ConfluenceQA()
    confluence_qa.init_embeddings()
    confluence_qa.init_models()
    confluence_qa.retrieval_qa_chain()
    st.session_state["confluence_qa"] = confluence_qa

# @st.cache_resource
def load_confluence(config) -> None:
    confluence_qa = st.session_state.get("confluence_qa")
    
    confluence_qa.vector_db_confluence_docs(config)

with st.sidebar.form(key='cf-params'):
    st.markdown('## Add your configs')
    confluence_url = st.text_input("paste the confluence URL", "https://templates.atlassian.net/wiki/")
    username = st.text_input(label="confluence username",
                             help="leave blank if confluence page is public"
                             )
    space_key = st.text_input(label="confluence space",
                              help="Space of Confluence",
                              value="RD")
    page_id = st.text_input(label="page id",
                            help="ID of Confluence page to ingest. Leave blank to ingest all pages in space"
                            )
    api_key = st.text_input(label="confluence api key",
                            help="leave blank if confluence page is public",
                            type="password")

    submitted1 = st.form_submit_button(label='Submit')

    st.session_state["config"] = {
        "confluence_url": confluence_url,
        "page_id": page_id if page_id != "" else "None",
        "username": username if username != "" else None,
        "api_key": api_key if api_key != "" else None,
        "space_key": space_key
    }
    
    if submitted1:
        with st.spinner(text="Ingesting Confluence..."):
            ### Hardcoding for https://templates.atlassian.net/wiki/ and space RD to avoid multiple OpenAI calls.
            config = st.session_state["config"]
            # if config["confluence_url"] == "https://templates.atlassian.net/wiki/" and config["space_key"] == "RD":
            #     config["persist_directory"] = ".chroma_db"
            st.session_state["config"] = config

            load_confluence(st.session_state["config"])
        st.write("Confluence Space Ingested")
    # else:
    #     st.write("Restored previously stored Confluence data")

st.title("Confluence Q&A")

question = st.text_input('Ask a question', placeholder="What's the most common cause of ELK issues?")

if st.button('Get Answer', key='button2'):
    with st.spinner(text="Asking LLM..."):
        confluence_qa = st.session_state.get("confluence_qa")
        if not question:
            st.write("Please enter a question.")
        elif not confluence_qa:
            st.write("Please load Confluence page first.")
        else:
            sources = []
            sources_str = ""
            answer, sources = confluence_qa.answer_confluence(question)
            citations = []
            for i, source in enumerate(sources, start=1):
                citations.append(f"[{i}]({source})")
            citations_str = ", ".join(citations)

            st.session_state["chat_history"] += f"You: {question}\nBot: {answer}\n\nSources:\n{citations_str}\n================================\n"
                            
            output = f"{answer}\n\nSources:\n{citations_str}"
                                    
            st.write(output)
        
            
# Display chat history
st.text_area("Chat History", st.session_state["chat_history"], height=600)