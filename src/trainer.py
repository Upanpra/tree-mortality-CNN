import torch
from typing import Optional
from torch import nn

SUPPORTED_LR_SCHEDULERS = {"cosine"}


def trainCNN(net, train_loader, test_loader,
          num_epochs, learning_rate, modelsavepath,
          compute_accs=True, lr_scheduler: Optional[str] = "cosine"):

    # classification called the Cross Entropy Loss and the popular Adam optimizer.
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    #optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate) #, weight_decay = 1e-2)

    if lr_scheduler is not None:
        assert lr_scheduler in SUPPORTED_LR_SCHEDULERS, f"{lr_scheduler} unsupported. Not in {SUPPORTED_LR_SCHEDULERS}"
        if lr_scheduler == "cosine":
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
        else:
            raise NotImplementedError(f"lr scheduler {lr_scheduler} not in supported set {SUPPORTED_LR_SCHEDULERS}")

    train_accs = []
    test_accs = []

    best_acc = 0
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader): # Loop over each batch in train_loader    
            # If you are using a GPU, speed up computation by moving values to the GPU
            if torch.cuda.is_available():
                net = net.cuda()
                images = images.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()               # Reset gradient for next computation
            outputs = net(images)               # Forward pass: compute the output class given a image
            loss = criterion(outputs, labels)   # Compute loss: difference between the pred and true
            loss.backward()                     # Backward pass: compute the weight
            optimizer.step()                    # Optimizer: update the weights of hidden nodes

            if (i + 1) % 100 == 0:  # Print every 100 batches
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}')
        if compute_accs: 
          train_acc = accuracy(net, data_loader= train_loader)
          test_acc = accuracy(net, data_loader= test_loader)
          train_accs.append(train_acc)
          test_accs.append(test_acc)
          print(f'Epoch [{epoch + 1}/{num_epochs}], Train Acc {100 * train_acc:.2f}%, Test Acc {100 * test_acc:.2f}%')
          if test_acc > best_acc:
              best_acc = test_acc
              torch.save(net.state_dict(), modelsavepath)

        if lr_scheduler is not None:
            lr_scheduler.step()
    if compute_accs:
        return net, train_accs, test_accs
    else:
        return net


def accuracy(net, data_loader):
    net.eval()
    correct = 0
    total = 0
    for images, labels in data_loader:
        if  torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()

        outputs = net(images)                           # Make predictions
        _, predicted = torch.max(outputs.data, 1)       # Choose class with highest scores
        total += labels.size(0)                         # Increment the total count
        correct += (predicted == labels).sum().item()   # Increment the correct count
    net.train()
    return correct / total
