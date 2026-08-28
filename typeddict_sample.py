from typing import TypedDict

class Person(TypedDict):

    name: str
    age: int


new_person:Person = {'name': 'Tushar' , 'age': 20}

print(new_person)