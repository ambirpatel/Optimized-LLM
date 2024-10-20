# Optimized-LLM

# Vicuna 7B Model Serving with FastAPI and Dynamic Batching

This repository provides an implementation of a FastAPI-based server to serve a **compiled Vicuna-7B model** using **Llama.cpp**. The server implements **dynamic batching** to efficiently manage multiple concurrent requests, reducing the overall response time by grouping requests into a single batch before sending them to the model for inference.

---

### Model Compilation Steps

To use the **compiled Vicuna-7B** model, follow these steps to compile the model using **llama.cpp**:

1. **Download the Base Model**:
   - Obtain the base Vicuna-7B model from the LMSYS repository.

2. **Install Llama.cpp**:
   - Clone the llama.cpp repository and install the required dependencies.
   - Llama.cpp supports quantization to reduce memory usage and offers a simple API for loading and running models.
   - It's compatible with various Llama-based models, including Vicuna.

3. **Convert Base Model to GGUF Format**:
   - Convert the downloaded model to the GGUF format, which is more optimized for llama.cpp inference.

4. **Quantize the GGUF Model**:
   - Use the quantization feature in llama.cpp to reduce the model size for faster inference and lower memory consumption.

   **Memory/Disk Requirements** (for the 7B Vicuna model):
   - **Original size**: 13 GB
   - **Quantized size (Q4_K_M)**: 3.9 GB

---

## FastAPI Server with Dynamic Batching

The FastAPI server is designed to handle multiple incoming requests, batch them together, and perform inference on the compiled Vicuna model in an optimized manner. This ensures efficient GPU usage and reduces overall latency by processing several requests in a single forward pass.

### Features

- **Dynamic Batching**: Collects multiple requests within a short window (e.g., 50ms) and sends them in one batch to the model for inference.
- **Efficient Memory Management**: Takes advantage of llama.cpp's ability to load quantized models, reducing memory footprint.
- **Concurrency Handling**: Manages multiple requests asynchronously using FastAPI and `asyncio` for high throughput.

### Prerequisites

1. **Vicuna Model**: Ensure you have a quantized version of the Vicuna-7B model (in GGUF format) compiled with llama.cpp.
2. **Python Libraries**:
   - `llama_cpp`
   - `fastapi`
   - `uvicorn`
   - `aiohttp`

### Installation

1. Install the necessary Python libraries:
   ```bash
   pip install fastapi uvicorn llama_cpp aiohttp
   ```

2. Place the **quantized Vicuna model** in a directory (e.g., `./quantized_model/vicuna_7b_FP16_K_M.gguf`).

---

### Code Overview

#### FastAPI Server Code with Dynamic Batching

The FastAPI app runs a server that handles incoming requests, batches them, and sends them to the model for inference.

- **Dynamic Batching**: Requests are batched dynamically based on the configured batch size and timeout values (`BATCH_SIZE`, `BATCH_TIMEOUT`).
- **Async Processing**: The server uses asynchronous request handling to maximize concurrency.
- **Vicuna Inference**: The compiled Vicuna model is loaded using the `llama_cpp.Llama` interface.


### Running the Server

To start the FastAPI server, run the following command:

```bash
uvicorn your_fastapi_script:app --reload
```

This will serve the model at `http://localhost:8000/generate/`.

---

## Testing Dynamic Batching

You can test the dynamic batching feature by sending multiple requests concurrently. The following script uses `asyncio` and `aiohttp` to send requests in parallel to the FastAPI server.

### Test Script

After running the FastAPI server, execute the test script with:

```bash
python test_batch_inference.py
```

The script sends multiple requests concurrently and checks whether the server batches them. The responses should indicate that batching is working correctly, and the total time taken should reflect the efficiency gains from batching.

---