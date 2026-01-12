from fastapi import FastAPI
import uvicorn
from mangum import Mangum
from src.services import SubmitQueryRequest, answer_extraction

app = FastAPI()
handler = Mangum(app)

@app.get("/")
def index():
    return {"Hello": "World"}

@app.post("/submit_query")
async def submit_query_endpoint(request:SubmitQueryRequest):
    """ Endpoint to submit a query for processing."""
    result_json = await answer_extraction(request)   
    return result_json
if __name__ == "__main__":
    # Run this as a server directly.
    port = 8080
    print(f"Running the FastAPI server on port {port}.")
    uvicorn.run(app, host="0.0.0.0", port=port)