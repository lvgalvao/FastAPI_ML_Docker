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