from langchain_community.document_loaders import WebBaseLoader
from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

model = ChatGoogleGenerativeAI(model = 'gemini-3.5-flash-lite' , max_output_tokens = 100, thinking_level='minimal')

prompt = PromptTemplate(
    template='answer the {question} from the following : {text}',
    input_variables=['question' , 'text']
)

parser = StrOutputParser()

chain = prompt | model | parser

loader = WebBaseLoader('https://en.wikipedia.org/wiki/Machine_learning')

docs = loader.load()

result = chain.invoke({'question': 'what is machine learning' , 'text': docs[0].page_content})

print(result)