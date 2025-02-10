import torch
import grpc
import split_lora_pb2
import split_lora_pb2_grpc
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load GPT-2 (Client Side)
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
client_layers = nn.ModuleList(model.transformer.h[:6])  # First 6 layers

# Prepare Input
text = "Hello, how are you?"
input_ids = tokenizer.encode(text, return_tensors="pt")

# Forward Pass (Client Side)
embeddings = model.transformer.wte(input_ids)
for block in client_layers:
    embeddings = block(embeddings)[0]

# Send to Server
channel = grpc.insecure_channel("SERVER_IP:50051")  # Replace SERVER_IP
stub = split_lora_pb2_grpc.SplitLoraStub(channel)
request = split_lora_pb2.EmbeddingsRequest(embeddings=embeddings.flatten().tolist())
response = stub.SendEmbeddings(request)

# Receive Logits
print("Received logits:", response.logits)
