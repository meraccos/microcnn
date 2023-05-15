from graphviz import Source

def to_dot(node, dot=None):
    from graphviz import Digraph

    if dot is None:
        dot = Digraph()
        dot.attr(rankdir='RL')  # sets the direction of the graph to left-right
        dot.node(name=str(id(node)), label=f"{node.data:.3f}\n{node.grad:.3f}\n{node.op}")

    if node.children:
        for child in node.children:
            dot.node(name=str(id(child)), label=f"{child.data:.3f}\n{child.grad:.3f}\n{child.op}")
            dot.edge(str(id(node)), str(id(child)))
            to_dot(child, dot)

    return dot