from graphviz import Digraph


def train(model, X, y, n_epochs, loss_fc, optimizer):
    for epoch in range(n_epochs):
        model_out = model.forward(X)

        loss = loss_fc.forward(model_out[0], y)

        loss.zero_grad()
        loss.backward(grad=1.0)

        optimizer.forward(model.parameters())

        if epoch % 10 == 0:
            print(
                f"# Epoch: {epoch},   loss: {loss.data:.3f},   out: {model_out[0].data:.3f}"
            )


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
