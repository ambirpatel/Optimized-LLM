import asyncio
import aiohttp
import time

# The URL of the FastAPI server
API_URL = "http://localhost:8000/generate/"

# Test prompts to send to the server
test_prompts = [
    "Tell me a story about a dragon.",
    "What is the capital of France?",
    "Who won the 2018 FIFA World Cup?",
    "Explain quantum physics in simple terms.",
    # "What is the meaning of life?",
    # "Describe the history of the Roman Empire.",
    # "How do airplanes fly?",
    # "What is the Python programming language used for?",
    # "What is the future of AI?",
    # "Give me a recipe for pancakes."
]

# Async function to send a POST request with a prompt to the API
async def send_request(prompt: str):
    async with aiohttp.ClientSession() as session:
        payload = {"prompt": prompt}
        async with session.post(API_URL, json=payload) as response:
            assert response.status == 200
            data = await response.json()
            print(f"Prompt: {prompt}\nResponse: {data['response']}\n")
            return data['response']

# Function to test batching by sending multiple requests concurrently
async def test_batch_inference():
    start_time = time.time()

    # Schedule requests for all test prompts
    tasks = [send_request(prompt) for prompt in test_prompts]

    # Run the tasks concurrently and gather results
    await asyncio.gather(*tasks)

    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")

# Entry point to start the test
if __name__ == "__main__":
    asyncio.run(test_batch_inference())
