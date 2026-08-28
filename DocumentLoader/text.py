from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

model = ChatGoogleGenerativeAI(model = 'gemini-3.5-flash-lite' , max_output_tokens = 100, thinking_level='minimal')

prompt = PromptTemplate(
    template='write a two line summary of the poem : {poem}',
    input_variables=['poem']
)

parser = StrOutputParser()

chain = prompt | model | parser

loader = TextLoader('cricket.txt', encoding='utf-8')

docs = loader.load()

print(chain.invoke({'poem' : docs[0].page_content}))