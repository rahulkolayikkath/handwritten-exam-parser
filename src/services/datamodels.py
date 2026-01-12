"""" 
This module contains all data models 
"""
from pydantic import BaseModel
from typing import List 


class Point(BaseModel):
    x: int
    y: int

class BoundingBox(BaseModel):
    p1: Point
    p2: Point
    p3: Point
    p4: Point

class MolmoResponse(BaseModel):
    bbox: List[BoundingBox]


# class to submit the input 
class SubmitQueryRequest(BaseModel):
    pdf_url_path : str = "default-url"

# extraction structure for structured response from Gemini
extraction_structure = {
  "type": "object",
  "properties": {
    "student_id": {
      "type": "string",
      "description": "Extract the student id. if not filled, return unknown"
    },
    "student_name": {
      "type": "string",
      "description": "Extract the student name"
    },
    "page_no": {
      "type": "string",
      "description": "Extract the page no"
    },
    "question_numbers": {
        "type": "array",
        "description": "Extract the question ids seen on the left margin.In the order, top to bottom",
        "items": {
        "type": "string"
      }
    },
    "starts_with_continuation": {
        "type": "string",
        "description": "true, if the first line of content on the page is not indicated with a question number to its left margin. false, if the first line of content on the page is indicated with an question number to the left margin. "
    }
  },
  "required": [
    "student_id",
    "student_name",
    "page_no",
    "question_numbers",
    "starts_with_continuation",
  ]
}
