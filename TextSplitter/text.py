from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

text = '''
Artificial Intelligence (AI) is a branch of computer science that enables machines to perform tasks that normally require human intelligence. Machine Learning (ML) is a subset of AI that allows computers to learn from data and improve their performance without being explicitly programmed for every task. AI and ML are used in applications such as chatbots, recommendation systems, self-driving cars, image recognition, and healthcare
'''
#for differnt types of text we can use .from language and pass language = Language.Markdown for eg
splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
)

result =splitter.split_text(text)

print(result)