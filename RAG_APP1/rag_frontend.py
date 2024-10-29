import streamlit as st
import rag_backend as demo 

st.set_page_config(page_title="Sample HR Chatbot")

new_title= "HR Q & A with RAG"
st.markdown(new_title,unsafe_allow_html=True)


if 'vector_index' not in st.session_state:
    with st.spinner("Waiting......."):
        st.session_state.vector_index=demo.hr_index()


input_text= st.text_area("Input text", label_visibility="collapsed")
go_button=st.button("Submit Question", type="primary")

if go_button:
    with st.spinner("Waiting......"):
        response_content=demo.hr_rag_response(index=st.session_state.vector_index,question=input_text)
        st.write(response_content)

#  To run the application , hit the below command
#  streamlit run rag_frontend.py  