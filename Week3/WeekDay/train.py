# 패키지 임포트 
import torch
import torch.nn as nn


from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

# haper-parameters 설정
## 모델관련
input_size=28*28
hidden_size=500
output_size=10
## 데이터 관련
batch_size=100
## 학습 관련
lr = 0.001
device='cuda' if torch.cuda.is_available() else 'cpu'

## 데이터 업로드 ## 
# MNIST dataset 모듈을 불러오기
train_dataset = MNIST(root='../../data', train=True, download=True , transform=ToTensor())
test_dataset = MNIST(root='../../data', train=False, download=True , transform=ToTensor())
# dataloader 모듈 만들기
train_loader = DataLoader(dataset = train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)   
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

## Ai 모델 설계 ##
class mlp(nn.Module):

## 모델 설계도 만들기 
    def __init__(self, input_size=28*28, hidden_size=500, output_size=10):
        super().__init__()
        # 4개 nn.Linear 이용 (전체 입력: 28*28, Hidden layer 수 :3, 출력 : 10)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)

    def forward(self, x): # x : 데이터 (이미지, batch_size x channel x height x with)
        batch_size, channel, height, width = x.shape
        
        
        # 데이터 펼치기 -> fc1 -> fc2 -> fc3 -> fc4 -> 출력
        x = x.reshape(batch_size, height*width)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x
        

# 모델 객체 생성
model = mlp(input_size, hidden_size, output_size).to(device)

# loss 함수 설정 (분류 문제, classification -> cross-entropy)
critera = CrossEntropyLoss()

# optimizer 설정 (Adam)
optim = Adam(params=model.parameters(), lr=lr)
print('dd')

## for loop 문 ##
for idx, (data, label) in enumerate(train_loader):
    # 불러온 데이터를 모델에 넣기
    output = model(data)

    # 나온 출력으로 loss계산
    loss = critera(output, label)
    # loss로 back prop 진행
    loss.backward()
    # optimizer를 이용해 최적화 진행
    optim.step()
    optim.zero_grad()

    if idx % 100 == 0 :

        print('loss의 값은 : ', loss.item())
    # 학습 중간에 평가를 진행해서
    # 성능이 좋으면 저장
