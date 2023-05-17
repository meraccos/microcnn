from graphviz import Digraph
import random
from microcnn.nn import Softmax
from graphviz import Source

def train(model, X_all, y_all, n_epochs, batch_size, loss_fc, optimizer, grad_clip=1.0):
    n_samples = len(X_all)
    for epoch in range(n_epochs):
        # Randomly permute the indices of the data points
        indices = list(range(n_samples))
        random.shuffle(indices)

        for i in range(0, n_samples, batch_size):
            # Get the indices for this batch
            batch_indices = indices[i:i+batch_size]
            
            X_batch = [X_all[idx] for idx in batch_indices]
            y_batch = [y_all[idx] for idx in batch_indices]

            # Perform forward and backward pass for each data point in batch
            batch_loss = 0.0
            for X, y in zip(X_batch, y_batch):
                model_out = model.forward(X)
                loss = loss_fc.forward(model_out, y)
                batch_loss += loss.data
                
                model.zero_grad()
                loss_fc.backward()
                model.backward()

            # Average the gradients
            for param in model.parameters():
                param.grad /= len(batch_indices)

            # Clip the gradients to prevent explosion
            for param in model.parameters():
                param.grad = max(min(param.grad, grad_clip), -grad_clip)

            # Update the parameters
            optimizer.step(model.parameters(), lr=3e-3)

            # Clear the gradients for the next iteration
            for param in model.parameters():
                param.grad = 0

        if epoch % 100 == 0:
            avg_loss = batch_loss / len(batch_indices)
            print(
                f"# Epoch: {epoch},   average loss: {avg_loss:.3f}"
            )




   
        # Randomly pick the batch of training data
        t_data_idx = random.sample(list(range(len(X_all))),
                                   k=min(len(X_all), batch_size), )
        
        X_train = [X_all[index] for index in t_data_idx]
        y_train = [y_all[index] for index in t_data_idx]
        
        epoch_loss = 0.0
        eval = 0
        eval_probs = []
        found = []
        
        for X, y in zip(X_train, y_train):  #X = [1.0], y = [0.8]
            model_out = model.forward(X)
            
            loss = loss_fc.forward(model_out, y)
            
            model.zero_grad()
            
            loss_fc.backward()
            model.backward()
            
            epoch_loss += loss.data

        for param in model.parameters():
            param.grad /= len(X_train)
            param.grad = max(min(param.grad, grad_clip), -grad_clip)

        optimizer.step(model.parameters(), lr=3e-3)
        
        for param in model.parameters():
            param.grad = 0

        if epoch % 100 == 0:
            avg_loss = epoch_loss / len(X_train)
            print(
                f"# Epoch: {epoch},   average loss: {avg_loss:.3f}, average accuracy: {eval} / {len(X_train)}, probs: {eval_probs} found values: {found}"
            )
    return model, loss


# def train(model, X_all, y_all, n_epochs, batch_size, loss_fc, optimizer, grad_clip=1.0):
#     for epoch in range(n_epochs):
#         # Randomly pick the batch of training data
#         t_data_idx = random.sample(list(range(len(X_all))),
#                                    k=min(len(X_all), batch_size), )
        
#         X_train = [X_all[index] for index in t_data_idx]
#         y_train = [y_all[index] for index in t_data_idx]
        
#         epoch_loss = 0.0
#         eval = 0
#         eval_probs = []
#         found = []
        
#         for X, y in zip(X_train, y_train):  #X = [1.0], y = [0.8]
#             model_out = model.forward(X)
#             model_out = Softmax().forward(model_out)
            
#             preds = [val.data for val in model_out]
#             prob = 0
#             pred = None
            
#             for index, val in enumerate(preds):
#                 if val > prob:
#                     prob = val
#                     pred = index
            
#             if y[pred] == 1:
#                 print(pred)
#                 eval += 1
#                 eval_probs.append(prob)
#                 found.append(pred)
            
#             # output = [round(val.data,2) for val in model_out]
#             # print('model out:', output)
#             # print('expected:', y)
#             # print('input: ', X)
#             print(len(model.parameters()))
            
#             loss = loss_fc.forward(model_out, y)
#             loss.grad = 1.0
#             print('backwarding')
#             loss.backward()
            
#             epoch_loss += loss.data

#         for param in model.parameters():
#             param.grad /= len(X_train)
#             param.grad = max(-grad_clip, min(param.grad, grad_clip))

#         grads = [param.grad for param in model.parameters()]

#         print('max grad: ', max(grads))
#         print('min grad: ', min(grads))
#         print('ave grad: ', sum(grads) / len(grads))

#         optimizer.step(model.parameters(), lr=3e-3)
        
#         for param in model.parameters():
#             param.grad = 0

#         if epoch % 1 == 0:
#             avg_loss = epoch_loss / len(X_train)
#             print(
#                 f"# Epoch: {epoch},   average loss: {avg_loss:.3f}, average accuracy: {eval} / {len(X_train)}, probs: {eval_probs} found values: {found}"
#             )
#     return model, loss


def to_dot(node, dot=None):
    if dot is None:
        dot = Digraph()
        dot.attr(rankdir="RL")  # sets the direction of the graph to right-left
        dot.node(
            name=str(id(node)), label=f"{node.data:.3f}\ngrad:{node.grad:.3f}\n{node.op}"
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
