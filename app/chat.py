# Chat interface logic
import streamlit as st
import requests

def _append_message(role: str, text: str):
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    st.session_state["messages"].append({"role": role, "text": text})

def render_chat(api_url: str, create_lead_url: str):
    st.header("Chat")
    chat_col, side_col = st.columns([3,1])

    with chat_col:
        if "messages" not in st.session_state:
            st.session_state["messages"] = [
                {"role": "assistant", "text": "Hello — ask me about ESILV programs, admissions, courses or upload documents for context."}
            ]

        for msg in st.session_state["messages"]:
            if msg["role"] == "user":
                st.markdown(f"**You:** {msg['text']}")
            else:
                st.markdown(f"**Assistant:** {msg['text']}")

        user_input = st.text_input("Your question", key="chat_input")
        submit = st.button("Send")
        if submit and user_input:
            _append_message("user", user_input)
            # call API
            try:
                r = requests.post(api_url, json={"message": user_input}, timeout=300)
                r.raise_for_status()
                payload = r.json()
                answer = payload.get("answer", "No answer returned.")
                _append_message("assistant", answer)
                # if API asks for collecting lead, show small form
                if payload.get("action") == "collect_lead":
                    with st.form("lead_form", clear_on_submit=False):
                        st.write("I can collect your contact information.")
                        name = st.text_input("Full name")
                        email = st.text_input("Email")
                        interest = st.text_input("Interest (optional)")
                        sub = st.form_submit_button("Submit contact info")
                        if sub and name and email:
                            # submit to API
                            rr = requests.post(create_lead_url, data={"name": name, "email": email, "interest": interest})
                            if rr.ok:
                                st.success("Thanks — contact saved.")
                            else:
                                st.error("Failed to save contact.")
            except Exception as e:
                _append_message("assistant", "Service error: " + str(e))

    with side_col:
        st.markdown("### Conversation tools")
        if st.button("Clear chat"):
            st.session_state["messages"] = []
        st.markdown("Upload relevant documents in the Upload tab to improve answers.")