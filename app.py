import random
import streamlit as st
import time

from chat import load_model, get_response
def response_generator(prompt):
    response = get_response(model, all_words, tags, prompt, intents)

    for word in response.split():
        yield word + " "
        time.sleep(0.1)

model, all_words, tags, intents = load_model("data.pth")



def main():
    st.title("Simple Clothes Store Chatbot")

    #Set up chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []


    #Display chat history
    with st.container():
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    #Get user input
    if prompt := st.chat_input("Have a question? Ask me!"):
        st.session_state.messages.append({"role": "user", "content": prompt})

        #Display message in chat
        with st.chat_message("user"):
            st.markdown(prompt)

        #Display response in chat
        with st.chat_message("assistant"):
            response = response_generator(prompt)
            response = st.write_stream(response)

        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()