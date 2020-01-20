import foolbox
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

class MnistModel(nn.Module):
    def __init__(self):
        super(MnistModel, self).__init__()
        # mnist의 경우 28*28의 흑백이미지(input channel=1)이다.
        self.conv1 = nn.Conv2d(1, 32, kernel_size = 5, padding=2)
        # feature map의 크기는 14*14가 된다
        # 첫번재 convolution layer에서 나온 output channel이 32이므로 2번째 input도 32
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 5, padding=2)
        # feature map의 크기는 7*7이 된다
        # fc -> fully connected, fc는 모든 weight를 고려해서 만들기 때문에 cnn에서는 locally connected를 이용하여 만든다.
        # nn.Linear에서는 conv를 거친 feature map을 1차원으로 전부 바꿔서 input을 한다. 이게 64*7*7
        self.fc1 = nn.Linear(64*7*7, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 64*7*7) # linear에 들어갈 수 있도록 reshape
        x = F.relu(self.fc1(x)) # fully connected에 relu 적용
        x = F.dropout(x, training=self.training) # 가중치 감소만으로는 overfit을 해결하기가 어려움, 그래서 뉴런의 연결을 임의로 삭제
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def DeepFoolAttack(model, image, label):
    attack = foolbox.attacks.DeepFoolAttack(model)
    return attack(image, label)

def train(model, device, train_loader, optimizer, epoch, log_interval, train_loader_origin):
    model.train()
    avg_loss = 0
    # in training loop:
    for batch_idx, (data, target) in enumerate(train_loader):
        # gpu나 cpu로 데이터를 옮김
        data, target = data.to(device), target.to(device)
        # gradient descent전에 gradient buffer를 0으로 초기화 하는 역할을 한다
        output = model(data)
        # negative log-likelihood: nll loss, 딥러닝 모델의 손실함수이다
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step() # Does the update based on the current gradient stored in grad
        # 여기서 기존에 저장되어있던 gradient를 기반으로 update 하기 때문에 위의 초기화가 필요함
        avg_loss+=F.nll_loss(output, target, reduction='sum').item()
        
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader_origin.dataset),
                100. * batch_idx / len(train_loader_origin), loss.item()))
    avg_loss/=len(train_loader_origin.dataset)
    return avg_loss

is_cuda = torch.cuda.is_available()
device = torch.device('cuda' if is_cuda else 'cpu')

model = MnistModel()
pretrained_model = './data/mnist_um.pth'
model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))

fmodel = foolbox.models.PyTorchModel(model, bounds=(0, 1), num_classes=10)

mnist_transform = transforms.Compose([
    transforms.ToTensor(), 
])

train_batch_size=64
test_batch_size=1
epochs=30
lr = 0.01
momentum = 0.5
seed = 1
log_interval = 1000

optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

download_path = './data'
train_dataset = MNIST(download_path, transform=mnist_transform, train=True, download=True)
test_dataset = MNIST(download_path, transform=mnist_transform, train=False, download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=train_batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=test_batch_size,
                                          shuffle=False)

perturbed_images = []

print("Attack image!")
for image, label in train_loader:
    perturbed_np = DeepFoolAttack(fmodel, image.numpy(), label.numpy())
    perturbed = torch.from_numpy(perturbed_np)
    perturbed_images.append((perturbed, label))
print("Attack done!")


train_losses = []
test_losses = []
accuracy_list = []
for epoch in range(1, epochs + 1):
    trn_loss = train(model, device, perturbed_images, optimizer, epoch, log_interval, train_loader)
    #test_loss,accuracy = test(model, device, test_loader)
    train_losses.append(trn_loss)
    #test_losses.append(test_loss)
    #accuracy_list.append(accuracy)

PATH = './data/mnist_deepfool_model.pth'
torch.save(model.state_dict(), PATH)