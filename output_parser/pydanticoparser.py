from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task= "text-generation",
    pipeline_kwargs=dict(
        temperature = 0.5,
        max_new_tokens = 100
    )
)
model = ChatHuggingFace(llm = llm)

class Person(BaseModel):
    name: str = Field(description="Name of the person")
    age: int = Field(gt =18, description="age of the perosn")

parser = PydanticOutputParser(pydantic_object= Person)

template1 =PromptTemplate(
    template="Write a fictional name of a person with age{age}",
    input_variables=['age'],
    # partial_variables={'format_instruction': parser.get_format_instructions()}
)

chain = template1 | model | parser

result = chain.invoke({'age': 19})
print(result.content)
