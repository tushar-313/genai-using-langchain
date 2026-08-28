from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal
load_dotenv()

model = ChatGoogleGenerativeAI(model = 'gemini-3.6-flash' , max_new_tokens = 10)
class Review(TypedDict):
    summary: str
    sentiment: Annotated[Literal["pos", "neg", "neutral"] , "return the sentiment of this review either negative, positive or neutral"]
    pros: Annotated[Optional[list[str]] , "write all the pros in the review"]

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("the overall experience with this brand is bad it was too much slow and not accurate")

print(result)