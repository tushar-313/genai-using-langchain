from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

model = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

prompt = PromptTemplate(
   template= "Generate 2 facts on {topic}" ,
   input_variables= ['topic']
)

parser = StrOutputParser()

chain=  prompt | model | parser

result = chain.invoke({"topic": "cricket"})

print(result)

chain.get_graph().print_ascii()