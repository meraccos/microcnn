from graphviz import Digraph
import random
from microcnn.nn import Softmax

def train(model, X_all, y_all, n_epochs, batch_size, loss_fc, optimizer):
    for epoch in range(n_epochs):
        # Randomly pick the batch of training data
        t_data_idx = random.sample(list(range(len(X_all))),
                                   k=min(len(X_all), batch_size), )
        
        X_train = [X_all[index] for index in t_data_idx]
        y_train = [y_all[index] for index in t_data_idx]
        
        epoch_loss = 0.0
        for X, y in zip(X_train, y_train):  #X = [1.0], y = [0.8]
            model_out = model.forward(X)
            # model_out = Softmax.forward(model_out)
            

            loss = loss_fc.forward(model_out, y)
            loss.backward(grad=1.0)
            
            epoch_loss += loss.data

        # print(model.parameters())
        optimizer.step(model.parameters(), lr=1e-2)
        # print(model.parameters())
        # print(model.layers[0].parameters())
        loss.zero_grad()
        # print(model.parameters())

        if epoch % 1 == 0:
            avg_loss = epoch_loss / len(X_train)
            print(
                f"# Epoch: {epoch},   average loss: {avg_loss:.3f}"
            )
    return model, loss


def to_dot(node, dot=None):
    if dot is None:
        dot = Digraph()
        dot.attr(rankdir="RL")  # sets the direction of the graph to right-left
        dot.node(
            name=str(id(node)), label=f"{node.data:.3f}\n{node.grad:.3f}\n{node.op}"
        )

    if node.children:
        for child in node.children:
            dot.node(
                name=str(id(child)),
                label=f"{child.data:.3f}\n{child.grad:.3f}\n{child.op}",
            )
            dot.edge(str(id(node)), str(id(child)))
            to_dot(child, dot)

    return dot
