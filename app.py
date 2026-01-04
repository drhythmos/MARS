import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnablePassthrough
from processing import process_uploaded_file, add_to_vector_store, vector_store, retrieve_from_vector_store


class HFChatLLM(Runnable):
    def __init__(self, model, token):
        from huggingface_hub import InferenceClient
        self.client = InferenceClient(api_key=token)
        self.model = model

    def invoke(self, input, config=None, **kwargs):
        input_data = input

        if isinstance(input_data, dict):
            prompt_text = input_data.get("question") or str(input_data)
        else:
            prompt_text = str(input_data)

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt_text}],
            max_tokens=512,
            temperature=0.2,
        )
        return completion.choices[0].message.content


@st.cache_resource
def load_rag_chain(_cache_buster: str = "v2"):
    retriever = retrieve_from_vector_store()

    llm = HFChatLLM(
        model="Qwen/Qwen2.5-1.5B-Instruct",
        token=st.secrets["HUGGINGFACEHUB_API_TOKEN"],
    )

    prompt_template = """
    Answer the user's question based only on the provided context.
    Cite the source file from the metadata.

    Context:
    {context}

    Question:
    {question}
    """
    prompt = PromptTemplate.from_template(prompt_template)
    
    def format_docs(docs):
        return "\n\n".join(f"Source: {doc.metadata['source']}\nContent: {doc.page_content}" for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

rag_chain = load_rag_chain("v2")


# UI
st.title("Multimodal RAG MVP")
st.write("Upload files and chat with your data!")

# Upload Section
uploaded_file = st.file_uploader("Upload image/audio/pdf", type=["png", "jpg", "pdf", "wav"])
if uploaded_file:
    with st.spinner("Processing your file..."):
        data_dict, filename = process_uploaded_file(uploaded_file)
        add_to_vector_store(data_dict, filename)
    st.success(f"Added '{filename}' to FAISS vector store!")

st.divider()

# Chat Section
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_question := st.chat_input("What's up?"):
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = rag_chain.invoke(user_question)
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})