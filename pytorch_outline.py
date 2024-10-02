# Simple outline for model development and experiments

# IMPORTS
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

import torch.utils.data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# LOAD DATASET
# batch_size = 64
# num_classes = 10
# train_dataset = torchvision.datasets.MNIST(root = './data',
#                                            train = True,
#                                            transform = transforms.Compose([
#                                                   transforms.Resize((32,32)),
#                                                   transforms.ToTensor(),
#                                                   transforms.Normalize(mean = (0.1307,), std = (0.3081,))]),
#                                            download = True)
# test_dataset = torchvision.datasets.MNIST(root = './data',
#                                           train = False,
#                                           transform = transforms.Compose([
#                                                   transforms.Resize((32,32)),
#                                                   transforms.ToTensor(),
#                                                   transforms.Normalize(mean = (0.1325,), std = (0.3105,))]),
#                                           download=True)
# train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
#                                            batch_size = batch_size,
#                                            shuffle = True)
# test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
#                                            batch_size = batch_size,
#                                            shuffle = True)

# MODEL
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # build architecture here

    def forward(self, x):
        # activations between layers here
        return x 


# HYPERPARAMETERS
learning_rate = 0.016 
num_epochs = 10

model = Net()
print(model)

cost = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# TRAIN
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):  
        images = images.to(device)
        labels = labels.to(device)
        
        #Forward pass
        outputs = model(images)
        loss = cost(outputs, labels)
        	
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        		
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
        		           .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# save the final model after all epochs
torch.save(model.state_dict(), 'final_model.pth')

# TEST
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))
