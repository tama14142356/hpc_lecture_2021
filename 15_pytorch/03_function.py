import torch


class ReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input


torch.manual_seed(0)
epochs = 300
batch_size = 32
D_in = 784
H = 100
D_out = 10
learning_rate = 1.0e-06

# create random input and output data
x = torch.randn(batch_size, D_in)
y = torch.randn(batch_size, D_out)

# randomly initialize weights
w1 = torch.randn(D_in, H, requires_grad=True)
w2 = torch.randn(H, D_out, requires_grad=True)

for epoch in range(epochs):
    # forward pass: compute predicted y
    relu = ReLU.apply

    h = x.mm(w1)
    h_r = relu(h)
    y_p = h_r.mm(w2)

    # compute and print loss
    loss = (y_p - y).pow(2).sum()
    print(epoch, loss.item())

    # backward pass
    loss.backward()

    with torch.no_grad():
        # update weights
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        # initialize weights
        w1.grad.zero_()
        w2.grad.zero_()
