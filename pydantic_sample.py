from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    name: str = 'NO Name' #default value
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(gt = 0 , lt = 10, default= 9 , description='the aggregate grades of the student')

new_stu = {'email': 'aff1@gamil.com', 'cgpa':10}

student = Student(**new_stu)
print(student)