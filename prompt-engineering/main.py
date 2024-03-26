from langchain.prompts import PromptTemplate
from langchain.llms.openai import OpenAI
from langchain.chains import LLMChain
import os

os.environ.get["OPENAI_API_KEY"]
dummy_template = '''I want you to act as a acting doctor for people. In an easy way, explain the medical condition called {medical_condition}.'''

prompt = PromptTemplate(
    input_variables = ['medical_condition'],
    template= dummy_template
)

llm = OpenAI(temperature = 0.8)
chain1 = LLMChain(llm = llm, prompt=prompt)

response=chain1.run("malaria")
print(response)
 
