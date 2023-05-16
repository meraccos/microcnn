from utils import to_dot, train
from microcnn import nn

import torchvision.datasets as datasets

mnist_trainset = datasets.MNIST(root='./data', 
                                train=True, 
                                download=True, 
                                transform=None)

t_data_idx = list(range(5))

img_data_list = [list(mnist_trainset[i][0].getdata()) for i in t_data_idx]
img_labels = [mnist_trainset[i][1] for i in t_data_idx]
img_labels_10 = [[1.0 if i == label else 0 for i in range(10)] for label in img_labels]

img_data_scaled = [[pixel / 255.0 - 0.5 for pixel in image] for image in img_data_list ]

model = nn.Model(
    [
        nn.Layer(n_inputs=784, n_neurons=16, act_fn=nn.Tanh()),
        nn.Layer(n_inputs=16, n_neurons=16, act_fn=nn.Tanh()),
        nn.Layer(n_inputs=16, n_neurons=10, act_fn=nn.ReLU()),
    ]
)


# loss_fc = nn.RMSLoss()
loss_fc = nn.CrossEntropyLoss()
optimizer = nn.SGD()

train(model=model, 
      X_all=img_data_scaled, 
      y_all=img_labels_10, 
      n_epochs=55, 
      batch_size = 5, 
      loss_fc=loss_fc, optimizer=optimizer)
