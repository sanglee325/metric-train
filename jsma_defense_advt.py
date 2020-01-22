import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import transforms
from torchvision.datasets import MNIST
import torchvision.models as models

import numpy as np
import matplotlib.pyplot as plt

from advertorch.attacks import JacobianSaliencyMapAttack as JSMA
from torchvision.utils import save_image

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

def jsma_attack(device, loader):
    print("Attack Image!!")
    
    umodel = MnistModel().to(device)
    umodel.eval()
    adversary = JSMA(umodel, num_classes=10, clip_min=0.0, clip_max=1.0,
                        loss_fn=None, theta=1.0, gamma=0.145, comply_cleverhans=False)
    perturbed_images = []
    for idx, (image, label) in enumerate(loader):
        print(idx)
        image, label = image.to(device), label.to(device)
        perturbed_image = adversary.perturb(image, label)
        perturbed_images.append((perturbed_image, label))
        if idx < 10:
            title = str(idx) + '.png'
            save_image(image, title)

    print("Attack Done!!")
    return perturbed_images


def advtrain(model, device, train_loader, optimizer, epoch, log_interval, org):
    model.train()
    avg_loss = 0
    # in training loop:

    for batch_idx, (data, target) in enumerate(train_loader):
        # gpu나 cpu로 데이터를 옮김
        data, target = data.to(device), target.to(device)
        # gradient descent전에 gradient buffer를 0으로 초기화 하는 역할을 한다
        optimizer.zero_grad()
        output = model(data)
        # negative log-likelihood: nll loss, 딥러닝 모델의 손실함수이다
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step() # Does the update based on the current gradient stored in grad
        # 여기서 기존에 저장되어있던 gradient를 기반으로 update 하기 때문에 위의 초기화가 필요함
        avg_loss+=F.nll_loss(output, target, reduction='sum').item()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(org.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    avg_loss/=len(org.dataset)
    return avg_loss

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    # .no_grad는 required_grad = True여도 gradient 계산을 하지 않고 더 빠르게 넘어갈 수 있도록 한다
    # training이 아니기 때문에 gradient 계산을 할 필요가 없음
    for idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
        # dim 1에서 가장 큰 값을 가지는 것의 index를 반환
        pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
        # pred.eq(data)는 pred와 data 배열이 일치하는지를 검사
        # target을 pred처럼 보이게 만들어서 .sum()을 이용해 일치하는 것들의 개수를 구한다
        # .item()은 tensor의 값을 스칼라로 만드는 역할을 한다
        correct += pred.eq(target.view_as(pred)).sum().item()


    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader),
        100. * correct / len(test_loader)))
    accuracy = 100. * correct / len(test_loader)
    return test_loss,accuracy
                                  
train_batch_size=64
test_batch_size=1
epochs=10
lr = 0.01
momentum = 0.5
seed = 1
log_interval = 1000

is_cuda = torch.cuda.is_available()
device = torch.device('cuda' if is_cuda else 'cpu')
if is_cuda :
    print("CUDA available!")

pretrained_model = './model/mnist_um.pth'
model = MnistModel().to(device)
model.load_state_dict(torch.load(pretrained_model, map_location=device))

# momentum: 기울기에서 속도의 개념을 추가, 기울기 업데이트시 폭을 조절한다
# lr: learning rate
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

# mean=0.5 std=1.0으로 Normalize한다
mnist_transform = transforms.Compose([
    transforms.ToTensor(), 
])

download_path = './data'
train_dataset = MNIST(download_path, transform=mnist_transform, train=True, download=True)
test_dataset = MNIST(download_path, transform=mnist_transform, train=False, download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=train_batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=test_batch_size,
                                          shuffle=True)


perturbed_test = jsma_attack(device, test_loader)
perturbed_train = jsma_attack(device, train_loader)

train_losses = []
test_losses = []
accuracy_list = []

for epoch in range(1, epochs + 1):
    #trn_loss = advtrain(model, device, perturbed_train, optimizer, epoch, log_interval, train_loader)
    #train_losses.append(trn_loss)
    test_loss,accuracy = test(model, device, perturbed_test)
    test_losses.append(test_loss)
    accuracy_list.append(accuracy)

PATH = './model/mnist_jsma_adv_model.pth'
torch.save(model.state_dict(), PATH)