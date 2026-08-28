from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
load_dotenv()

doc1 = Document(
    page_content="Virat Kohli is an Indian cricketer known for his excellent batting.",
    metadata={"team": "Royal Challengers Bengaluru"}
)

doc2 = Document(
    page_content="Rohit Sharma is an Indian cricketer and one of the leading opening batsmen.",
    metadata={"team": "Mumbai Indians"}
)

doc3 = Document(
    page_content="MS Dhoni is a former Indian captain known for his finishing and wicketkeeping skills.",
    metadata={"team": "Chennai Super Kings"}
)

doc4 = Document(
    page_content="Sachin Tendulkar is a legendary Indian cricketer.",
    metadata={"team": "Mumbai Indians"}
)

doc5 = Document(
    page_content="Jasprit Bumrah is an Indian fast bowler.",
    metadata={"team": "Mumbai Indians"}
)

docs = [doc1, doc2, doc3, doc4, doc5]

vector_store = Chroma(
    embedding_function= GoogleGenerativeAIEmbeddings(model='gemini-embedding-001'),
    persist_directory='chroma_db',
    collection_name='sample'
)
# vector_store.add_documents(docs)

# print(vector_store.similarity_search_with_score(
#     query='who among these are a bowler?', 
#     k =2
# ))

# print(vector_store.similarity_search_with_score(
#     query='.', 
#     filter={"team": "Mumbai Indians"}
# ))

update_doc = Document(
    page_content='rohit sharma is bad cricketer and dump player in ipl',
    metadata={'team' : 'RCB'}
)
# vector_store.update_document(document_id='47e9c592-0846-45cd-9679-1482086f2eb4' , document=update_doc)

vector_store.delete(ids='47e9c592-0846-45cd-9679-1482086f2eb4')

print(vector_store.get(include=['embeddings', 'documents', 'metadatas']))