from graphviz import Digraph
# from graphviz import Source
import random
import torchvision.datasets as datasets
from microcnn.value import Value


def train(model, X_all, y_all, n_epochs, batch_size, loss_fc, optimizer, grad_clip=1.0):
    n_samples = len(X_all)
    for epoch in range(n_epochs):
        # Randomly permute the indices of the data points
        indices = random.sample(range(n_samples), n_samples)

        epoch_loss = 0.0
        correct = 0

        for i in range(0, n_samples, batch_size):
            # Get the indices for this batch
            batch_indices = indices[i : i + batch_size]

            X_batch = [X_all[idx] for idx in batch_indices]
            y_batch = [y_all[idx] for idx in batch_indices]

            # Zero out the gradients from the previous batch
            model.zero_grad(batch=False)

            # Perform forward and backward pass for each data point in batch
            for X, y in zip(X_batch, y_batch):
                model_out = model.forward(X)
                loss = loss_fc.forward(model_out, y)
                epoch_loss += loss.data

                # Count number of correct predictions
                model_out_data = [val.data for val in model_out]
                y_data = [val.data for val in y]

                model_pred = max(enumerate(y_data), key=lambda pair: pair[1])[0]
                y_gt = max(enumerate(model_out_data), key=lambda pair: pair[1])[0]

                if model_pred == y_gt:
                    correct += 1

                loss_fc.backward()
                model.backward()
                
                model.zero_grad(batch=True)

            # Average and clip the gradients
            for param in model.parameters():
                param.grad /= len(batch_indices)
                param.grad = max(min(param.grad, grad_clip), -grad_clip)

            # Update the parameters
            optimizer.step(model.parameters())

        avg_loss = epoch_loss / n_samples
        accuracy = correct / n_samples
        print(
            f"Epoch {epoch+1}/{n_epochs}, Average Loss: {avg_loss:.3f}, Accuracy: {accuracy*100:.2f}%\n"
        )

    return model, loss


def test(model, X_test, y_test):
    n_samples = len(X_test)
    correct = 0

    for X, y in zip(X_test, y_test):
        model_out = model.forward(X)
        model_out_data = [val.data for val in model_out]
        y_data = [val.data for val in y]

        model_pred = max(enumerate(y_data), key=lambda pair: pair[1])[0]
        y_gt = max(enumerate(model_out_data), key=lambda pair: pair[1])[0]

        if model_pred == y_gt:
            correct += 1

    accuracy = correct / n_samples
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    return accuracy


def mnist_data_retriever(data_idx_start, data_idx_end, train_bool):
    mnist_dataset = datasets.MNIST(root='./data', 
                                train=train_bool, 
                                download=True, 
                                transform=None)

    data_idx = list(range(data_idx_start, data_idx_end))

    images = [list(mnist_dataset[i][0].getdata()) for i in data_idx]
    labels = [mnist_dataset[i][1] for i in data_idx]

    labels_base10 = [[Value(1.0, op='in') if i == label else Value(0.0, op='in') 
                    for i in range(10)] for label in labels]

    images_scaled = [[Value(pixel * 2 / 255.0 - 1.0, op='in') 
                        for pixel in image] for image in images]
    
    return images_scaled, labels_base10


def to_dot(node, dot=None):
    if dot is None:
        dot = Digraph()
        dot.attr(rankdir="RL")  # sets the direction of the graph to right-left
        dot.node(
            name=str(id(node)),
            label=f"{node.data:.3f}\ngrad:{node.grad:.3f}\n{node.op}",
        )

    if node.children:
        for child in node.children:
            dot.node(
                name=str(id(child)),
                label=f"{child.data:.3f}\ngrad:{child.grad:.3f}\n{child.op}",
            )
            dot.edge(str(id(node)), str(id(child)))
            to_dot(child, dot)

    return dot


# dot = to_dot(loss)
# Source(dot.source)
