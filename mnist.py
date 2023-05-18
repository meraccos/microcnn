from utils import train, test, mnist_data_retriever
from microcnn import nn
import pickle

# train_images, train_labels = mnist_data_retriever(0, 6000, train_bool=True)
# test_images, test_labels = mnist_data_retriever(6000, 10000, train_bool=False)


train_images, train_labels = mnist_data_retriever('fashion', 0, 6000, train_bool=True)
test_images, test_labels = mnist_data_retriever('fashion', 6000, 9900, train_bool=False)

model = nn.Model(
    [
        nn.Linear(n_inputs=784, n_neurons=8, act_fn=nn.ReLU),
        # nn.Linear(n_inputs=16, n_neurons=16, act_fn=nn.Tanh),
        nn.Linear(n_inputs=8, n_neurons=10, act_fn=nn.Identity),
    ]
)

loss_fc = nn.SoftmaxCrossEntropyLoss(l1_coeff=5e-4, l2_coeff=5e-4)
optimizer = nn.Adam(lr=1e-3)

for epoch in range(50):
    test(model, test_images, test_labels)
    train(
        model=model,
        X_all=train_images,
        y_all=train_labels,
        n_epochs=1,
        batch_size=5,
        loss_fc=loss_fc,
        optimizer=optimizer,
    )

    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
