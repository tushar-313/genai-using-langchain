from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

model = ChatGoogleGenerativeAI(model = 'gemini-3.6-flash' , max_completion_token=10)

message = [
    SystemMessage(content='You are a helpful assistent'),
    HumanMessage(content="explain me about langchain")
]
result = model.invoke(message)
message.append(AIMessage(content=result.text))
print(message)