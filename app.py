import streamlit as st
from qa import load_llm, ask_question, remove_think_tags

st.title("📄 Document Question Answering")
query = st.text_input("Ask a question:")

if "llm" not in st.session_state:
    st.session_state.llm = load_llm()

if query:
    answer_dict = ask_question(query, st.session_state.llm)
    answer = answer_dict.get("result", "")
    clean_output = remove_think_tags(answer)
    st.write("**Answer:**", clean_output)
