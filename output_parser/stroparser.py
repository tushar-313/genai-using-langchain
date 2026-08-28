from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task= "text-generation",
    pipeline_kwargs=dict(
        temperature = 0.5,
        max_new_tokens = 100
    )
)
model = ChatHuggingFace(llm = llm)

template1 =PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=["topic"]
)
template2 = PromptTemplate(
    template= "write a 5 line summary of the following text\n{text}",
    input_variables=["text"]
)

# prompt1 = template1.invoke({'topic': 'floods'})
# print(prompt1)

parser = StrOutputParser()

chain = template1 | model | template2 | model | parser

result = chain.invoke({'topic': 'black hole'})
print(result)
