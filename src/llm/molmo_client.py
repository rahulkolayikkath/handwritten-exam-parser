"""Async Client implementation of molmo
Note: This is a standalone class and doesnot inherit from class in base.py
"""
import asyncio
from typing import Optional, List
import os 
import requests 
import json
import regex as re
from pydantic import BaseModel
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

# utils for molmo 
# class molmo_client
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

def get_coords(output_string, image_shape):
    """
    Function to get x, y coordinates given Molmo model outputs.
    :param output_string: Output from the Molmo model.
    :param image: Image in PIL format.
    Returns:
        coordinates: Coordinates in format of [(x, y), (x, y)]
    """
    h, w = image_shape
    
    if 'points' in output_string:
        matches = re.findall(r'(x\d+)="([\d.]+)" (y\d+)="([\d.]+)"', output_string)
        coordinates = [(int(float(x_val)/100*w), int(float(y_val)/100*h)-50) for _, x_val, _, y_val in matches]
    else:
        match = re.search(r'x="([\d.]+)" y="([\d.]+)"', output_string)
        if match:
            coordinates = [(int(float(match.group(1))/100*w), int(float(match.group(2))/100*h)-50)]
        else:
            coordinates = None
    if coordinates:
        coordinates.sort(key=lambda xy: xy[1])
    return coordinates

def extrapolte_cords(cords, image_shape) -> MolmoResponse:
    if cords is None:
        return [] 
    height, width = image_shape
    bbox_list = []

    for i, (x, y) in enumerate(cords):
        # ---- A1 ----
        A1 = Point(x=50, y=y)

        # ---- A2 ----
        x_right = width - 200
        A2 = Point(x=x_right, y=y)

        # ---- A3 & A4 ----
        if i + 1 < len(cords):
            _, next_y = cords[i+1]
            A3 = Point(x=50, y=next_y - 10)
        else:
            A3 = Point(x=50, y=height - 50)

        A4 = Point(x=x_right, y=A3.y)

        bbox_list.append(BoundingBox(p1=A1, p2=A2, p3=A3, p4=A4))
    return bbox_list

class MolmoAsyncClient():
    """Client for Molmo model via RunPod async API"""
    def __init__(self, poll_interval: float = 5.0, endpoint_id:Optional[str] = None, api_key:Optional[str] = None):
        """ 
        Args:
        endpoint_id : Runpod endpoint_id
        api_key : Runpod API key 
        poll_interval: Seconds to wait between polling 
        """

        endpoint_id = endpoint_id or os.environ.get("ENDPOINT_ID")
        if not endpoint_id:
            raise ValueError(
                "The Endpoint ID must be provided either as an argument or via environment variable"
            )
        api_key = api_key or os.environ.get("RUNPOD_API_KEY")
        if not api_key:
            raise ValueError(
                "The RUNPOD_API_KEY must be provided either as an argument or via environment variable"
            )
        
        super().__init__()
        self.endpoint_id = endpoint_id
        self.api_key = api_key
        self.poll_interval = poll_interval
        
    
    def _submit_job(self, input_data):
        endpoint_url = f"https://api.runpod.ai/v2/{self.endpoint_id}/run"
        headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {self.api_key}"}
        payload = {"input": input_data}
        response = requests.post(endpoint_url, headers=headers, json=payload)
        job_id = response.json()['id']
        return job_id
    
    def _retry_job(self, job_id:str):
        """Retry a failed or timed-out RunPod job."""
        endpoint_url = f"https://api.runpod.ai/v2/{self.endpoint_id}/retry/{job_id}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = requests.post(endpoint_url, headers=headers)
        return job_id

    async def _poll_job(self, job_id: str):
        """Poll the job status until completion or failure."""
        endpoint_url = f"https://api.runpod.ai/v2/{self.endpoint_id}/status/{job_id}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        while True:
            response = requests.get(endpoint_url, headers=headers)
            
            # Add error checking for polling request
            if response.status_code != 200:
                raise RuntimeError(f"Failed to poll job {job_id}: {response.text}")
            try:
                status_data = response.json()
            except json.JSONDecodeError:
                raise RuntimeError(f"Invalid JSON response for job {job_id}: {response.text}")
            
            # Check if 'status' key exists
            if 'status' not in status_data:
                raise RuntimeError(f"No 'status' key in response for job {job_id}: {status_data}")
                
            status = status_data['status']
            if status == "COMPLETED":
                return status_data["output"]
            elif status == "FAILED" or status == "TIMED_OUT":
                job_id = self._retry_job(job_id)
            else:
                await asyncio.sleep(self.poll_interval)  # Use async sleep
    
    async def generate(self,prompt: str,image_url: str, image_shape: tuple)-> MolmoResponse:
        input_data = {
        "image": image_url,
        "text": prompt}
        job_id = self._submit_job(input_data=input_data)
        response = await self._poll_job(job_id)
        output = response['output']
        bbox= extrapolte_cords(get_coords(output, image_shape), image_shape)
        return MolmoResponse(
            bbox=bbox
        )
