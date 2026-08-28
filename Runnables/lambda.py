from langchain_core.runnables import RunnableSequence, RunnableParallel , RunnableLambda, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = ChatGoogleGenerativeAI(
    model = 'gemini-3.5-flash-lite',
    max_output_tokens = 100,
    thinking_level = 'minimal'
)

def word_counter(text):
    return len(text.split())

prompt = PromptTemplate(
    template='write a one line joke on topic : {text}',
    input_variables=['text']
)
parser = StrOutputParser()

runnable_counter = RunnableLambda(word_counter)

joke_chain = RunnableSequence(prompt, model, parser)

parallel_chain =RunnableParallel({
    'joke' :  RunnablePassthrough(),
    'word_count': RunnableLambda(word_counter)
}
)

chain = RunnableSequence(joke_chain, parallel_chain)

print(chain.invoke({'text': 'Friend'}))

