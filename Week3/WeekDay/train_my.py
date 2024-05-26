# 패키지 임포트


# hyperparameter 설정
## 모델관련


## 데이터 관련


## optim 관련

# 데이터 업로드 
## MNIST dataset 모듈 불러오기

# dataloader 모듈 만들기


# AI 모델 설계


## 모델 설계도 만들기
    # Initialize 
    
        # 4개의 FC만들기 nn.Linear 이용(전체 입력 : 28*28, hidden layer 수= 3, 출력 =10)
       

    # 모델 순전파 설정
     # x: 데이터 (이미지, batch_size * channel * height * width)
       

        # 일렬로 데이터 펼치기 reshpae -> fc1 -> fc2 -> fc3 -> fc4 -> 출력
       

# model 객체 생성


# loss 함수 설정 (분류 문제, classification -> cross-entropy)


# optimizer 설정 (adam)


# for loop 문 생성

    # 불러온 data를 모델에 넣기
    
    
    # 나온 출력으로 loss 계산
    
    # loss로 backprop 진행
   
    # optmim 을 이용해 최적화 진행
 
    # 학습 중간에 평가를 진행해서
    # 성능이 좋으면 저장