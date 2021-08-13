import torch


def backward_hook(self, in_grad, out_grad):
    self.in_grad = in_grad
    self.out_grad = out_grad


x = torch.tensor([[1.], [2.]], requires_grad=True)
y = torch.tensor([[1.], [1.]])
model = torch.nn.Linear(1, 1, bias=False)
for param in model.parameters():
    torch.nn.init.constant_(param, 2)
model.register_full_backward_hook(backward_hook)
criterion = torch.nn.MSELoss(reduction='mean')
y_p = model(x)
loss = criterion(y_p, y)
print('loss   :', loss.data)
loss.backward()

print(len(model.out_grad))
print('dl/dy:', model.out_grad[0].data)
print('dl/dx:', model.in_grad[0].data)
for param in model.parameters():
    print('grad   :', param.grad.data)
