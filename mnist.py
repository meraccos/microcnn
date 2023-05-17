from utils import to_dot, train, test
from microcnn import nn
from microcnn.value import Value

import torchvision.datasets as datasets


######### Get the training data ready
mnist_trainset = datasets.MNIST(root='./data', 
                                train=True, 
                                download=True, 
                                transform=None)

t_data_idx = list(range(10000))

img_data_list = [list(mnist_trainset[i][0].getdata()) for i in t_data_idx]
img_labels = [mnist_trainset[i][1] for i in t_data_idx]


img_labels_10 = [[Value(1.0, op='in') if i == label else Value(0.0, op='in') 
                  for i in range(10)] for label in img_labels]

img_data_scaled = [[Value(pixel / 255.0, op='in') 
                    for pixel in image] for image in img_data_list ]



######### Get the test data ready
mnist_testset = datasets.MNIST(root='./data', 
                               train=False, 
                               download=True, 
                               transform=None)

t_data_idx_test = list(range(len(mnist_testset)))

img_data_list_test = [list(mnist_testset[i][0].getdata()) for i in t_data_idx_test]
img_labels_test = [mnist_testset[i][1] for i in t_data_idx_test]
img_labels_10_test = [[Value(1.0, op='in') if i == label else Value(0.0, op='in') 
                       for i in range(10)] for label in img_labels_test]

img_data_scaled_test = [[Value(pixel / 255.0 - 0.5, op='in') 
                         for pixel in image] for image in img_data_list_test]






model = nn.Model(
    [
        nn.Layer(n_inputs=784, n_neurons=16, act_fn=nn.LeakyReLU()),
        nn.Layer(n_inputs=16, n_neurons=16, act_fn=nn.LeakyReLU()),
        nn.Layer(n_inputs=16, n_neurons=10, act_fn=None),
    ]
)


# loss_fc = nn.RMSLoss()
# loss_fc = nn.CrossEntropyLoss()
loss_fc = nn.SoftmaxCrossEntropyLoss()
optimizer = nn.SGD()

test(model, img_data_scaled, img_labels_10)

train(model=model, 
      X_all=img_data_scaled, 
      y_all=img_labels_10, 
      n_epochs=100, 
      batch_size = 32, 
      loss_fc=loss_fc, optimizer=optimizer)
