import os
from PIL import Image
import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

class InputDataset:  #генератор датасета
    def __init__(self, set_dir, target_dir, Class): #папка с сетом-set_dir. названия изображений 
                                                    #должны быть соответствовать номеру строки в файле с метками
        self.set = set_dir
        self.Class = Class               
        imgs = sorted(os.listdir(self.set))
        with open(target_dir) as f:
            number = []
            for i in f:
                number.append(int(i))
        self.name = list(zip(imgs, number))

    def __getitem__(self, idx):
        name_img, num = list(zip(*self.name))

        img_path = os.path.join(self.set, name_img[idx])
        img = Image.open(img_path)
        img = ToTensor()(img)
        
        target = torch.zeros(self.Class, dtype=torch.float).scatter_(0, torch.tensor(num[idx]), value=1)

        return (img, target)

    def __len__(self):
        return len(self.name)

class network(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq1 = nn.Sequential(
           nn.Conv2d(3, 20, 3),
           nn.Dropout2d(0.2),
           nn.BatchNorm2d(20),
           nn.ReLU(),
           nn.MaxPool2d(2),
           nn.ZeroPad2d(5))
        self.seq2 = nn.Sequential(
           nn.Conv2d(20, 30, 3),
           nn.Dropout2d(0.2),
           nn.BatchNorm2d(30),
           nn.ReLU(),
           nn.MaxPool2d(2),
           nn.ZeroPad2d(5))
        self.seq3 = nn.Sequential(
           nn.Conv2d(30, 40, 2),
           nn.Dropout2d(0.2),
           nn.BatchNorm2d(40),
           nn.ReLU(),
           nn.MaxPool2d(2),
           nn.ZeroPad2d(3))
        self.seq4 = nn.Sequential(
           nn.Conv2d(40, 50, 2),
           nn.Dropout2d(0.2),
           nn.BatchNorm2d(50),
           nn.ReLU(),
           nn.MaxPool2d(2),
           nn.ZeroPad2d(3))
        self.seq5 = nn.Sequential(
           nn.Conv2d(50, 60, 2),
           nn.Dropout2d(0.2),
           nn.BatchNorm2d(60),
           nn.ReLU(),
           nn.MaxPool2d(2))
        self.dens = nn.Sequential(
           nn.Linear(36*60, 123),
           nn.Dropout1d(0.2),
           nn.BatchNorm1d(123),
           nn.Sigmoid(),
           nn.Linear(123, 10),
           nn.Softmax(dim=1))
    def forward(self, x):
        x = self.seq1(x)
        x = self.seq2(x)
        x = self.seq3(x)
        x = self.seq4(x)
        x = self.seq5(x)
        x = x.view(-1, 36*60)
        x = self.dens(x)
        return x

def train(p, loss_f, optimizer, model, loader):
    for epoch in range(1, p+1):
        loss_sum = 0
        for batch, (x, y) in enumerate(loader, 1):

            out = model(x.to(device))
            loss = loss_f(out, y.to(device))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_sum += loss.item()
            if batch%20 == 0:
                print(f'Средняя ошибка: {round(loss_sum/20, 4)} [{epoch}: {batch*len(x)}/{len(loader)*len(x)}]')
                loss_sum = 0
        #torch.save(model.state_dict(), 'путь куда хочешь сохранить')
    print('\n°¬° ❤️<^~^')


if __name__=='__main__':
    data = datasets.CIFAR10(  #датасет цветных картинок
       root="data",
       train=True,
       download=True,
       transform=ToTensor(),
       target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)))
    loader = torch.utils.data.DataLoader(data, batch_size=50, shuffle=True)  #генератор
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu' #подключить GPU, если имеется
    model = network().to(device)  #создаём объект нейросети
    loss = nn.CrossEntropyLoss()  #функция потерь
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  #оптимизатор Адам

    train(p=10, #обучение 
      loss_f=loss, 
      optimizer=optimizer,
      model=model,
      loader=loader)
