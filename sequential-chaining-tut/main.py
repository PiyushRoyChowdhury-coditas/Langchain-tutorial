import os
# from constants import openai_key
from langchain.llms.openai import OpenAI
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain

os.environ.get["OPENAI_API_KEY"]
st.title('Langchain Demo')
input_text = st.text_input("Search the topic you want")

# prompt templates
first_input_prompt = PromptTemplate(
    input_variables=['name'],
    template="Tell me about the celebrity {name}"
) 

llm = OpenAI(temperature=0.8)
chain = LLMChain(llm=llm, prompt=first_input_prompt,verbose=True, output_key='person')

first_input_prompt = PromptTemplate(
    input_variables=['person'],
    template="When was the {person} born"
)

chain2 = LLMChain(llm=llm, prompt=first_input_prompt,verbose=True, output_key='dob')

parent_chain=SimpleSequentialChain(chains = [chain,chain2],verbose=True)


if(input_text):
    # st.write(llm(input_text))
    # st.write(chain.run(input_text))
    st.write(parent_chain.run(input_text)) 