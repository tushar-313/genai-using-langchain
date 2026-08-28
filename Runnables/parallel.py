from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableSequence



model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    max_output_tokens=500,
    thinking_level = "minimal"
)
prompt1 = PromptTemplate(
    template='Write a tweet about {text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template='write a one line about {text}',
    input_variables=['text']
)

parser = StrOutputParser()

chain = RunnableParallel(
    {
        'tweet' : RunnableSequence(prompt1, model,  parser),
        'post' : RunnableSequence(prompt2, model, parser)
    }
)

print(chain.invoke({'text' : 'web dev'}))