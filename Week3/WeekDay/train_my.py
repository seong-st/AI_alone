# 패키지 임포트
import torch
import torch.nn as nn

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

device = 'cuda' if torch.cuda.is_available() else 'cpu'  
print(device)

# hyperparameter 설정
## 모델관련
input_size = 28*28
hidden_size = 500
output_size = 10

## 데이터 관련
batch_size = 100

## optim 관련
lr = 0.001

# 데이터 업로드 
## MNIST dataset 모듈 불러오기
train_dataset = datasets.MNIST(root='../../data', train=True, download = True, transform = transforms.ToTensor())
test_dataset = datasets.MNIST(root='../../data', train=False, download = True, transform = transforms.ToTensor())

#image, label = train_dataset[1]

#plt.imshow(image.squeeze().numpy(), cmap='gray')

#plt.title('label = %s' %label)
#plt.show()

# dataloader 모듈 만들기
train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True) 
test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False)
# AI 모델 설계
class mlp(nn.Module):
    

## 모델 설계도 만들기
    # Initialize 
    def __init__(self, input_size = input_size, hidden_size = hidden_size, output_size = output_size): 
        super(mlp, self).__init__()
        # 4개의 FC만들기 nn.Linear 이용(전체 입력 : 28*28, hidden layer 수= 3, 출력 =10)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)   
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)

    # 모델 순전파 설정
     # x: 데이터 (이미지, batch_size * channel * height * width)
    def forward(self, x):   
        batch_size, channel, height, width = x.shape
        # 일렬로 데이터 펼치기 reshpae -> fc1 -> fc2 -> fc3 -> fc4 -> 출력
        x = x.reshape(batch_size, height*width)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x   

# model 객체 생성
model = mlp(input_size, hidden_size, output_size).to(device)

# loss 함수 설정 (분류 문제, classification -> cross-entropy)
criteria = CrossEntropyLoss()

# optimizer 설정 (adam)
optimizer = Adam(params=model.parameters(), lr = lr)

# for loop 문 생성
for idx, (data, label) in enumerate(train_loader):
    # 불러온 data를 모델에 넣기
    output = model(data)
    
    # 나온 출력으로 loss 계산
    loss = criteria(output, label)
    # loss로 backprop 진행
    loss.backward()
    # optmim 을 이용해 최적화 진행
    optimizer.step()
    optimizer.zero_grad()

    if idx % 100 == 0:
        print('loss의 값은 : ', loss.item())
    # 학습 중간에 평가를 진행해서
    # 성능이 좋으면 저장