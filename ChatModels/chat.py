from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.6-flash", max_completion_tokens =10)
result =model.invoke("poem on cricket 5 lines")
print(result.text)
                     