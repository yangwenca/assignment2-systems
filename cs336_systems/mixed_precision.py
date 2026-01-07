import torch


"""
mixed precision accumulation
tensor(10.0001)
tensor(9.9531, dtype=torch.float16)
tensor(10.0021)
tensor(10.0021)

loss of computation in fp16 is much smaller than that of accumulation in fp16
"""


"""
benchmark_mixed_precision
(a)
model parameters are fp32
output of first feed forward is fp16
output of layer norm is fp16 (internal fp32)
model's predicted logits is fp16
loss is fp32
gradient is fp32

(b)
sqrt
no, bf16 does not need to treat layernorm differently

(c)
mixed precision runs faster, most of parameters become bf16
model size is almost half
"""

s = torch.tensor(0,dtype=torch.float32)
for i in range(1000):
    s += torch.tensor(0.01,dtype=torch.float32)
print(s)

s = torch.tensor(0,dtype=torch.float16)
for i in range(1000):
    s += torch.tensor(0.01,dtype=torch.float16)
print(s)

s = torch.tensor(0,dtype=torch.float32)
for i in range(1000):
    s += torch.tensor(0.01,dtype=torch.float16)
print(s)

s = torch.tensor(0,dtype=torch.float32)
for i in range(1000):
    x = torch.tensor(0.01,dtype=torch.float16)
    s += x.type(torch.float32)
print(s)


class ToyModel(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_features, 10, bias=False)
        self.ln = torch.nn.LayerNorm(10)
        self.fc2 = torch.nn.Linear(10, out_features, bias=False)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        print(x.dtype)
        x = self.ln(x)
        print(x.dtype)
        x = self.fc2(x)
        return x


model = ToyModel(20, 100)
dtype = torch.float16
x = torch.rand((1000, 20))


with torch.autocast(device_type="cuda", dtype=dtype):
    y = model(x)
    print(y.dtype)
    print(model.parameters(), y)
