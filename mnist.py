from utils import train, test, mnist_data_retriever
from microcnn import nn

train_images, train_labels = mnist_data_retriever(0, 9000, train_bool=True)
test_images, test_labels = mnist_data_retriever(9000, 10000, train_bool=False)

model = nn.Model(
    [
        nn.Layer(n_inputs=784, n_neurons=16, act_fn=nn.Tanh),
        nn.Layer(n_inputs=16, n_neurons=16, act_fn=nn.Tanh),
        nn.Layer(n_inputs=16, n_neurons=10, act_fn=nn.Identity),
    ]
)

loss_fc = nn.SoftmaxCrossEntropyLoss()
# optimizer = nn.SGD()
optimizer = nn.Adam()

for epoch in range(50):
    test(model, test_images, test_labels)
    train(model=model, 
        X_all=train_images, 
        y_all=train_labels, 
        n_epochs=1, 
        batch_size = 5, 
        loss_fc=loss_fc, optimizer=optimizer)
    print()
