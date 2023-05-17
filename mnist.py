from utils import train, test, mnist_data_retriever
from microcnn import nn

train_images, train_labels = mnist_data_retriever(0, 1000, train_bool=True)
test_images, test_labels = mnist_data_retriever(1000, 1500, train_bool=False)

model = nn.Model(
    [
        nn.Layer(n_inputs=784, n_neurons=16, act_fn=nn.LeakyReLU()),
        nn.Layer(n_inputs=16, n_neurons=16, act_fn=nn.LeakyReLU()),
        nn.Layer(n_inputs=16, n_neurons=10, act_fn=None),
    ]
)

loss_fc = nn.SoftmaxCrossEntropyLoss()
optimizer = nn.SGD()

test(model, test_images, test_labels)

train(model=model, 
      X_all=train_images, 
      y_all=train_labels, 
      n_epochs=100, 
      batch_size = 5, 
      loss_fc=loss_fc, optimizer=optimizer)
