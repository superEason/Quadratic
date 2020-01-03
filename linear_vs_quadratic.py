
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from quadratic import quadratic
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import os

device = torch.device('cpu')


class QuadraticNeuralNet(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):
        super(QuadraticNeuralNet, self).__init__()
        self.fc1 = quadratic.Quadratic(input_size, num_classes)
        # self.relu = nn.ReLU()
        # self.fc2 = quadratic.Quadratic(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.fc1(x)
        # out = self.relu(out)
        # out = self.fc2(out)
        return out


class LinearNeuralNet(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):
        super(LinearNeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class pointDataset(Dataset):
    """Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Arguments:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """

    def __init__(self, data_tensor, label_tensor):
        assert data_tensor.size(0) == label_tensor.size(0)
        self.data = data_tensor
        self.label = label_tensor

    def __getitem__(self, index):
        return (data[index], label[index])

    def __len__(self):
        return self.data.size(0)


num = 500
input_size = 2
num_classes = 2
num_epochs = 1000
batch_size = 40
learning_rate = 0.01

theta = 2*math.pi*torch.rand(size=(num, 1))
X1 = torch.cat([(1 + 0.0 * torch.randn(num, 1)) * torch.cos(theta), (1 + 0.0*torch.randn(num, 1)) * torch.sin(theta)], axis=1)
X2 = torch.cat([(0.9 + 0.0 * torch.randn(num, 1)) * torch.cos(theta), (0.9 + 0.0 * torch.randn(num, 1)) * torch.sin(theta)], axis=1)

# a = torch.randint(0, 2, size = (num,1))
# X1 = torch.cat([a + 0.1 * torch.randn(num,1), 1-a + 0.1 * torch.randn(num,1)], axis = 1)
# X2 = torch.cat([a + 0.1 * torch.randn(num,1), a + 0.1 * torch.randn(num,1)], axis = 1)

data = torch.cat([X1, X2])
label = torch.cat((torch.zeros(num, dtype=int), torch.ones(num, dtype=int)))
mpl.use('Agg')
plt.figure()
plt.scatter(data[:, 0], data[:, 1], c=list(label))
Filename = 'Myfig.jpg'
plt.savefig(Filename)
os.system('sz '+Filename)

# model = LinearNeuralNet(input_size, 5, num_classes).to(device)
model = QuadraticNeuralNet(input_size, 1, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


dataset = pointDataset(data, label) 
train_dataset, test_dataset = torch.utils.data.random_split(dataset, (int(0.8*len(dataset)), int(0.2*len(dataset))))
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)


total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):  
        # Move tensors to the configured device
        images = images.reshape(-1, 2).to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item()))


with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 2).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the {} test images: {} %'.format(len(test_loader), 100 * correct / total))

for i in model.named_parameters():
    print(i)

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')




