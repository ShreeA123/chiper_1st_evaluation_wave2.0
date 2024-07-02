import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

torch.random.manual_seed(0)
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-128k-instruct",
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")

def generate_response(user_input):
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "assistant", "content": "Hello! I'm happy to assist you. I'm being developed by Team Cipher, a group of four members - [Shreebodh Inamdar], [Sayeed Amaan], [Pratik Chitti], and [Vaishnavi R B] - participating in a 24-hour hackathon. We're excited to create a helpful AI chat that can assist anyone from anywhere. Our goal is to provide useful information and answer questions to the best of our abilities. Please feel free to ask me any questions or provide feedback to help us improve!"},
        {"role": "user", "content": user_input},
    ]

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    generation_args = {
        "max_new_tokens": 500,
        "return_full_text": False,
        "temperature": 0.0,
        "do_sample": False,
    }

    output = pipe(messages, **generation_args)
    response = output[0]['generated_text']
    return response

st.title("Chatbot")
st.header("Talk to me!")

user_input = st.text_area("Type your message...", height=10)

if st.button("Send"):
    response = generate_response(user_input)
    st.text_area("Response:", value=response, height=10, disabled=True)

if st.button("Reset"):
    user_input = ""
    st.text_area("Type your message...", height=10, value="")