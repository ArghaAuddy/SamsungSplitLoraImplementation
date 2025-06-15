## Split LoRA Inference with gRPC & ProtoBuf
This project implements Split Inference using Low-Rank Adaptation (LoRA) on GPT-2, distributed between a client and server via gRPC and Protocol Buffers.

⚙️ Architecture:

Client:
Runs the first 6 layers of GPT-2
Sends intermediate embeddings over gRPC
Server:
Processes remaining layers
Returns logits


🔧 Tech Stack:

PyTorch, Transformers (Hugging Face)
gRPC + Protocol Buffers
Model: GPT2LMHeadModel

🌐 Use Case:

Ideal for resource-constrained edge devices, cloud offloading, and secure model partitioning.
