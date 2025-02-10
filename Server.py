import torch
import torch.nn as nn
import grpc
import split_lora_pb2
import split_lora_pb2_grpc
from concurrent import futures
import numpy as np
from transformers import GPT2LMHeadModel

# Load GPT-2 model (Server Side)
model = GPT2LMHeadModel.from_pretrained('gpt2')
server_layers = nn.ModuleList(model.transformer.h[6:])  # Last layers
linear = nn.Linear(768, model.config.vocab_size)

class SplitLoraServicer(split_lora_pb2_grpc.SplitLoraServicer):
    def SendEmbeddings(self, request, context):
        embeddings = torch.tensor(request.embeddings).view(1, -1, 768)  # Convert to tensor
        for block in server_layers:
            embeddings = block(embeddings)[0]
        logits = linear(embeddings).detach().numpy().flatten().tolist()
        return split_lora_pb2.LogitsResponse(logits=logits)

# Start gRPC Server
server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
split_lora_pb2_grpc.add_SplitLoraServicer_to_server(SplitLoraServicer(), server)
server.add_insecure_port("[::]:50051")
server.start()
print("Server is running on port 50051...")
server.wait_for_termination()
