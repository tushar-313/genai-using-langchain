from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableLambda, RunnableBranch
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    max_output_tokens=500,
    thinking_level = "minimal"
)


parser1 = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal['positive' , 'negative'] = Field(description='Give the sentiment of the feeback')

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template='Classify the sentiment into positive or negative \n{feedback}\n{format_instructions}',
    input_variables=['feedback'],
    partial_variables={'format_instructions' : parser2.get_format_instructions()}

)

prompt2 = PromptTemplate(
    template="write a one line positive response for the {feedback}",
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template="write a one line negative response for the {feedback}",
    input_variables=['feedback']
)

classify_chain = prompt1 | model | parser2

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == 'positive' , prompt2 | model | parser1),
    (lambda x: x.sentiment == 'negative' , prompt3 | model | parser1),
    RunnableLambda(lambda x: "could not find sentiment")
)

chain = classify_chain | branch_chain

result = chain.invoke({'feedback' : 'this is a very terrible phone'})

print(result)