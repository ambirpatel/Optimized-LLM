import asyncio
from fastapi import FastAPI, Request
from pydantic import BaseModel
import time

from llama_cpp import Llama


# Path to the model and loading the Vicuna model using llama_cpp
model_dir = "./quantized_model/"
model_path = model_dir + "vicuna_7b_FP16_K_M.gguf"
generation_kwargs = {
    "max_tokens": 100,
    "echo": False,
    "top_k": 1
}

# Initialize the model
try:
    model = Llama(model_path=model_path)
except:
    from huggingface_hub import snapshot_download

    model_name = "akp-usr/vicuna_compiled"
    snapshot_download(repo_id=model_name, local_dir=model_dir)
    model = Llama(model_path=model_path)

# FastAPI initialization
app = FastAPI()

# Dynamic batching configuration
BATCH_SIZE = 4  # Max number of requests to batch
BATCH_TIMEOUT = 0.05  # Timeout in seconds before processing batch (50ms)

# Shared queue for dynamic batching
request_queue = asyncio.Queue()

# Input data structure for FastAPI
class InputData(BaseModel):
    prompt: str

# Result dictionary to store responses for each request ID
results = {}


# Function to handle batched inference
async def process_batch():
    while True:
        # Collect requests for batching
        requests = []
        try:
            # Wait until we get enough requests or time out
            start_time = time.time()
            while len(requests) < BATCH_SIZE:
                try:
                    # Wait for new request (up to the batch timeout)
                    request = await asyncio.wait_for(request_queue.get(), timeout=BATCH_TIMEOUT)
                    requests.append(request)
                except asyncio.TimeoutError:
                    # Break the loop if timeout occurred
                    break

            if requests:
                # Prepare prompts for the batch
                prompts = [req["data"].prompt for req in requests]

                # Log batching info (optional)
                print(f"Processing batch of size: {len(prompts)}")

                # Model generation (batch inference)
                responses = [model(prompt, **generation_kwargs) for prompt in prompts]

                # Send results back to the requesters
                for i, req in enumerate(requests):
                    request_id = req["request_id"]
                    results[request_id] = responses[i]

                # Calculate and print batch processing time (optional)
                batch_time = time.time() - start_time
                print(f"Batch processed in {batch_time:.2f} seconds")

        except Exception as e:
            print(f"Error in batch processing: {e}")


# API endpoint to handle requests
@app.post("/generate/")
async def generate(data: InputData):
    # Generate a unique request_id for this request
    request_id = id(data)

    # Add the request to the shared queue
    await request_queue.put({"data": data, "request_id": request_id})

    # Wait until the result is ready (polling)
    while request_id not in results:
        await asyncio.sleep(0.01)

    # Retrieve and remove the result from the dictionary
    response = results.pop(request_id)
    return {"response": response}


# Startup event to launch the background batch processor
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(process_batch())


# Run the FastAPI app using Uvicorn (if needed for manual execution)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
