"""" 
This module contains all the helper functions.
"""
import os 
import boto3
from io import BytesIO
from PIL import Image
from .datamodels import Point, BoundingBox
from datetime import datetime
from collections import defaultdict
from config import system_prompts, format_user_prompt
from google import genai
import uuid
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

# Initialize S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.environ.get("S3_ACCESS_KEY_ID"),  # Ensure these env variables are set
    aws_secret_access_key=os.environ.get("S3_SECRET_ACCESS_KEY"),
    region_name=os.environ.get("S3_REGION")
)

def save_image_to_s3(snip: Image.Image, s3_key: str) -> str:
    """
    Save a PIL image to S3 and return its public URL.
    """
    bucket_name = os.environ.get("S3_BUCKET_NAME")
    
    # Create an in-memory file-like object to hold the image
    image_io = BytesIO()
    snip.save(image_io, format='JPEG')  # Save PIL image directly
    image_io.seek(0)  # Reset file pointer

    # Upload the image to S3
    s3_client.upload_fileobj(
        image_io,
        bucket_name,
        s3_key,
        ExtraArgs={'ContentType': 'image/jpeg'}
    )

    # Generate the S3 URL for the uploaded image
    region = os.environ.get("S3_REGION")
    image_url = f"https://{bucket_name}.s3.{region}.amazonaws.com/{s3_key}"

    return image_url

# utils for crop from bouding box 
def crop_bounding_box(img: Image.Image, bbox: BoundingBox) -> Image.Image:
    """
    Crop an axis-aligned bounding box from a PIL image.
    """
    # get min/max coordinates
    left = min(bbox.p1.x, bbox.p2.x, bbox.p3.x, bbox.p4.x)
    right = max(bbox.p1.x, bbox.p2.x, bbox.p3.x, bbox.p4.x)
    top = min(bbox.p1.y, bbox.p2.y, bbox.p3.y, bbox.p4.y)
    bottom = max(bbox.p1.y, bbox.p2.y, bbox.p3.y, bbox.p4.y)
    
    # crop image
    return img.crop((left, top, right, bottom))

# utils for combining extraction and layout data 
def combine_extraction_and_layout(extraction_list, bbox_list, image_list, image_shapes):
    """
    Combine extraction data and layout data into student-based structure.
    
    Args:
        extraction_list: list of dicts, each representing extraction for a page
        bbox_list: list of lists, each containing TextBlock objects for a page
    
    Returns:
        List of dicts, each representing a student and their combined page data
    """
    # Group by student_id and record details 
    # 
    students = {} 
    page = 1
    for extraction, bboxes, image, image_shape in zip(extraction_list, bbox_list, image_list, image_shapes):
        student_id = extraction.get("student_id")
        print(f"Running extraction of page{page}")
        page += 1
        if student_id not in students:
            students[student_id] = {
                "student_id": student_id,
                "student_name": extraction.get("student_name"),
                "page_numbers": [],
                "question_answered": [],
                "answers": defaultdict(list)
            }
        question_numbers = extraction.get("question_numbers")
        
        # Fixing Molmo Reponses 
        #1. if expected question numbers are empty discard any detected bouding boxes!
        if question_numbers == []:
            bboxes = []
        
        #2. check if the len(question_numbers) < len(bboxes). Molmo identified more bboxes
        # update bboxes with the merged fucntion
        if len(question_numbers) < len(bboxes):
            print("Molmo Failed! Merging with Gemini flash!")
            verification_prompt = format_user_prompt("verification_prompt", bboxes = bboxes, question_numbers = question_numbers)
            client = genai.Client(api_key = os.environ.get("GEMINI_API_KEY"))
            response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents= [image, verification_prompt],
            config={
                "response_mime_type": "application/json",
                "response_schema": list[BoundingBox],
            },)
            bboxes = response.parsed

        # check if the extraction contains "continuation"
        if extraction.get("starts_with_continuation") == "true":
            height, width = image_shape
            if bboxes != []:
                # create a defualt bounding box
                bbox = BoundingBox(p1=Point(x=50, y= 240), p2 = Point(x=width-200, y = 240), p3 = Point(x= 50, y= bboxes[0].p1.y- 10), p4 = Point(x= width-200, y= bboxes[0].p1.y- 10 ))
            else:
                bbox = BoundingBox(p1=Point(x=50, y = 240), p2 = Point(x= width-200, y = 240), p3 = Point(x = 50, y= height - 50), p4 = Point(x = width-200, y = height-50))
            
            snip = crop_bounding_box(image, bbox)
            if students[student_id]["question_answered"]:
                question_number = students[student_id]["question_answered"][-1] # last answered question
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                s3_key = f'{student_id}/{question_number}-cont-{timestamp}-{uuid.uuid4().hex}.jpg'
                path = save_image_to_s3(snip, s3_key)
                # add the image url to question path
                students[student_id]["answers"][question_number].append(path)
        
        # add page numbers 
        students[student_id]["page_numbers"].append(extraction.get("page_no"))

        # add question ids
        students[student_id]["question_answered"].extend(question_numbers)
        
        # Prevent bboxes out of index issue. 
        # A failing case handle where detected boudning boxes are less than numebr of questions expected, fill full page bouding boxes for all question answers
        if len(question_numbers) > len(bboxes):
            bboxes = [BoundingBox(p1=Point(x=50, y = 240), p2 = Point(x= width-200, y = 240), p3 = Point(x = 50, y= height - 50), p4 = Point(x = width-200, y = height-50)) for _ in question_numbers]
        
        # iterate through question numbers 
        for i, question_number in enumerate(question_numbers):
            bbox = bboxes[i]
            snip = crop_bounding_box(image, bbox)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            s3_key = f'{student_id}/{question_number}-{timestamp}-{uuid.uuid4().hex}.jpg' 
            path = save_image_to_s3(snip,s3_key)
            # add the question id 
            students[student_id]["answers"][question_number].append(path)
    
    #transform the answers section 
    for sid, student in students.items():
        answers_list = []
        for q_no, paths in student["answers"].items():
            answers_list.append({
                "question_no": str(q_no),
                "answerpath": paths
            })
        student["answers"] = answers_list
    
    return list(students.values())

import fitz  # PyMuPDF
from PIL import Image

def pdf_to_images(pdf_path, dpi=200):
    images = []
    doc = fitz.open(pdf_path)

    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # Render page to a pixmap (bitmap)
        mat = fitz.Matrix(dpi/72, dpi/72)  # scale for DPI
        pix = page.get_pixmap(matrix=mat)

        # Convert to PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)

    return images