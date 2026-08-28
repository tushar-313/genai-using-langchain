from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv

load_dotenv()


llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={
        "temperature": 0.5,
        "max_new_tokens": 150,
        "return_full_text": False
    }
)


model1 = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    max_output_tokens=500,
    thinking_level = "minimal"
)

model2 = ChatHuggingFace(llm=llm)


prompt1 = PromptTemplate(
    template="""
Generate a summary of the following text in exactly 2 short sentences.

Text:
{text}
""",
    input_variables=["text"]
)


prompt2 = PromptTemplate(
    template="""
Based on the text below, generate exactly ONE question and its short answer.

TEXT:
{text}

Return exactly in this format:

Question: <your question>
Answer: <your short answer>
""",
    input_variables=["text"]
)


prompt3 = PromptTemplate(
    template="""
Return the following information together exactly as provided.

Summary:
{notes}

Question and Answer:
{question}

Do not add a title, headings, introduction, conclusion, or any new information.
""",
    input_variables=["notes", "question"]
)


parser = StrOutputParser()


parallel_chain = RunnableParallel({
    "notes": prompt1 | model2 | parser,
    "question": prompt2 | model1 | parser
})


text = """
Cricket is one of the most popular sports in the world, especially in countries such as India, Australia, England, and Pakistan. It is played between two teams of eleven players, with the main objective of scoring more runs than the opposing team. The game requires a combination of batting, bowling, fielding, strategy, and teamwork. Cricket is played in different formats, including Test matches, One Day Internationals, and Twenty20 matches, each offering a different style and pace of play.
"""


merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain


print("PARALLEL OUTPUT:")
print(parallel_chain.invoke({"text": text}))

print("\nFINAL OUTPUT:")
result = chain.invoke({"text": text})
print(result)