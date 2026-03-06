from airllm import AutoModel

model = AutoModel.from_pretrained("meta-llama/Llama-3-70b")

output = model.generate("Explain AI simply")
print(output)