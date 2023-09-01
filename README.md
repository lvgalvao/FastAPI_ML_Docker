# FastAPI VQA (Visual Question-Answering) Project

## Introduction

The goal of this project is to refactor a starter file named `model_starter.py` and create an API using FastAPI for a Visual Question Answering (VQA) system. The system utilizes pretrained models from the HuggingFace Transformers library, specifically the `ViltProcessor` and `ViltForQuestionAnswering` models. Users can submit an image and a text-based question about the image, and the API will return an answer based on its analysis of both the image and text.

## Dependencies

* Python 3.x
* FastAPI
* Uvicorn
* Transformers
* Pillow

## You can install these dependencies via Poetry:

```bash
poetry install
```

## Via Docker
Alternatively, you can build and run the application using Docker:

```bash
Copy code
docker build -t fastapi-docker .
docker run -p 8000:8000 fastapi-docker
```

This will build a Docker image named fastapi-docker and then run it, mapping port 8000 in the container to port 8000 on your host machine.

## Project Structure

* `main.py`: Contains FastAPI routes and initializes the application.
* `model.py`: Handles model loading and defines the pipeline for prediction.
* `model_starter.py`: Original script to run predictions without the API.

## Usage

### Running the API

To run the API, navigate to the project directory in the terminal and execute:

```bash
uvicorn main:app --reload
```

This will start the FastAPI application, and it will be accessible at `http://127.0.0.1:8000`.

### API Endpoints

* `GET /`: Basic root endpoint that returns a Hello World message.
* `POST /ask`: Accepts an image file and a text query, and returns the answer to the question.

#### Sample Request for `/ask`

What is the brand of the drink?

![iFood Sample Image](/pic/ifood.jpg)


#### Sample Response

Coca-Cola


## Code Snippets

### main.py

```python
from model import model_pipeline
from fastapi import FastAPI, UploadFile
import io
from PIL import Image

app = FastAPI()

@app.post("/ask")
def ask(text: str, image: UploadFile):
    content = image.file.read()
    image = Image.open(io.BytesIO(content))
    result = model_pipeline(text, image)
    return {"answer": result}
```

### model.py

```python
from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image

processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

def model_pipeline(text: str, image: Image):
    encoding = processor(image, text, return_tensors="pt")
    outputs = model(**encoding)
    logits = outputs.logits
    idx = logits.argmax(-1).item()
    return model.config.id2label[idx]
```

### model_starter.py

```python
from transformers import ViltProcessor, ViltForQuestionAnswering
import requests
from PIL import Image

# prepare image + question
url = "https://static.ifood-static.com.br/image/upload/t_medium/pratos/ff7bf962-bccc-445c-9829-1f974561a4ff/202308230625_bojydjt65o.png"
image = Image.open(requests.get(url, stream=True).raw)
text = "What is the brand of the drink?"

processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

# prepare inputs
encoding = processor(image, text, return_tensors="pt")

# forward pass
outputs = model(**encoding)
logits = outputs.logits
idx = logits.argmax(-1).item()
print("Predicted answer:", model.config.id2label[idx])
```