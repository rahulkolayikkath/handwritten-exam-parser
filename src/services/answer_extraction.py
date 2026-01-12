""" This module ciontain the main fucntion for answer extraction"""
from .datamodels import SubmitQueryRequest, extraction_structure
import requests
from src.llm import GeminiAsyncClient, MolmoAsyncClient
import asyncio
import os
from config import system_prompts,format_user_prompt
import numpy as np
from datetime import datetime
import shutil
from .utils import save_image_to_s3, combine_extraction_and_layout, pdf_to_images
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

async def run_structured_inference(system_prompt, user_prompt, test_image, extraction_structure):
    extractor_model = GeminiAsyncClient(model="gemini-2.5-flash")
    return await extractor_model.generate_structured_response(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        image=test_image,
        structure=extraction_structure
    )
# code for inferenfce 
async def run_layout_inference(prompt, image_url, image_shape):
    molmo_model = MolmoAsyncClient()
    return  await molmo_model.generate(
        prompt = prompt,
        image_url= image_url,
        image_shape= image_shape
    )

async def answer_extraction(query:SubmitQueryRequest):
    ## convert the pdf url into images 
    # Lambda writable directory
    local_path = "/tmp/downloaded_pdfs"
    os.makedirs(local_path, exist_ok=True)  
    local_pdf_path = os.path.join(local_path, "downloaded_file.pdf")

    # download the pdf
    response = requests.get(query.pdf_url_path)
    with open(local_pdf_path, 'wb') as file:
        file.write(response.content)
    # convert pdf to images
    pdf_images = pdf_to_images(local_pdf_path)

    print("Downloaded and converted pdfs to images!!")
    # cleanup the entire folder
    if os.path.exists(local_path):
        shutil.rmtree(local_path) 

    print("Temporary PDF directory cleaned up!")
    
    # #testing using local pdf     
    # pdf_images = pdf_to_images("worksheet_test1.pdf")
    # print("loaded pdf from local!")
    
    ## Aysnc Gemini Extraction calls 
    #prepare promtps 
    page_extract_system_prompt = system_prompts["page_extract_prompt"]
    page_extract_user_prompt = format_user_prompt("page_extract_prompt")
    prompts = [(page_extract_system_prompt, page_extract_user_prompt, image, extraction_structure) for image in pdf_images]
    extraction_output = await asyncio.gather(*(run_structured_inference(*p) for p in prompts))
    print("Extraction from gemini complete!!")

    ## create alist of image urls and list of image shapes 
    pdf_image_shapes = [np.array(image).shape[:2] for image in pdf_images]
    pdf_image_paths = []
    for out,page in zip(extraction_output, pdf_images):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        s3_key = f'{out.structure["student_id"]}/{out.structure["page_no"]}-{timestamp}.jpg'
        path = save_image_to_s3(snip = page, s3_key= s3_key)
        pdf_image_paths.append(path)
    print("Uploaded all images to S3")

    ## Molmo bounding box detection 
    prompt_list = [format_user_prompt("molmo_extraction_prompt", question_numbers =  out.structure["question_numbers"]) for out in extraction_output]

    prompts = [(prompt, image_url, image_shape) for prompt , image_url, image_shape in zip(prompt_list, pdf_image_paths, pdf_image_shapes)]
    bbox_outputs = await asyncio.gather(*(run_layout_inference(*p) for p in prompts))
    print("Bounding box detection using molmo comlpted!")

    ##Define both outputs in usable form and combine
    extraction_list =  [extraction.structure  for extraction in extraction_output]
    bbox_list= [bbox.bbox for bbox in bbox_outputs] 
    student_data = combine_extraction_and_layout(extraction_list, bbox_list, pdf_images, pdf_image_shapes)
    print("successfully combined student data!")
    return student_data