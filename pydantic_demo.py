from pydantic import BaseModel,EmailStr,Field
from typing import Optional

class student(BaseModel):
    name: str="A"
    age:Optional[int] = None
    email: EmailStr
    cgpa:float=Field(gt=0,lt=4.0)


new_student = {'age':"100000", 'email':"avd@gmail.com", 'cgpa':3}

student=student(**new_student)
s_d=dict(student)

print(s_d['name'])
   