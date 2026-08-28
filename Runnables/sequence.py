from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence

prompt = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    max_output_tokens=500,
    thinking_level = "minimal"
)

prompt2 = PromptTemplate(
    template='explain the following joke {text}',
    input_variables=['text']
)

parser = StrOutputParser()

chain= RunnableSequence(prompt, model, parser , prompt2, model, parser) 

print(chain.invoke({'topic' : 'AI'}))