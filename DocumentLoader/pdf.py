from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('resume.pdf')

docs = loader.load()

print(len(docs))