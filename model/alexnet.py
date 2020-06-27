import torch.nn as nn
import model.Conv2d_quadratic as Cq
import torch.nn.functional as F

'''
modified to fit dataset size
'''
NUM_CLASSES = 10

# Origin
class AlexNet_0(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(AlexNet_0, self).__init__()
        self.features = nn.Sequential(
            # Cq.Conv2d_quadratic(3, 64, kernel_size=3, stride=2, padding=1),
            # nn.BatchNorm2d(64),
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # Cq.Conv2d_quadratic(64, 192, kernel_size=3, padding=1),
            # nn.BatchNorm2d(192),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # Cq.Conv2d_quadratic(192, 384, kernel_size=3, padding=1),
            # nn.BatchNorm2d(384),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Cq.Conv2d_quadratic(384, 256, kernel_size=3, padding=1),
            # nn.BatchNorm2d(256),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Cq.Conv2d_quadratic(256, 256, kernel_size=3, padding=1, bias=None),
            # nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.classifier(x)
        return x

class AlexNet_1(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(AlexNet_1, self).__init__()
        self.features = nn.Sequential(
            # Cq.Conv2d_quadratic(3, 64, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # Cq.Conv2d_quadratic(64, 192, kernel_size=3, padding=1),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # Cq.Conv2d_quadratic(192, 384, kernel_size=3, padding=1),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            # Cq.Conv2d_quadratic(384, 256, kernel_size=3, padding=1),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # Cq.Conv2d_quadratic(256, 256, kernel_size=3, padding=1, bias=None),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.classifier(x)
        return x

class AlexNet_2(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(AlexNet_2, self).__init__()
        self.features = nn.Sequential(
            Cq.Conv2d_quadratic(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            # nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            Cq.Conv2d_quadratic(64, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            # nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            Cq.Conv2d_quadratic(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            # nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            Cq.Conv2d_quadratic(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            # nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            Cq.Conv2d_quadratic(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            # nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.classifier(x)
        return x

class AlexNet_3(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(AlexNet_3, self).__init__()
        self.features = nn.Sequential(
            # Cq.Conv2d_quadratic(3, 64, kernel_size=3, stride=2, padding=1),
            # nn.BatchNorm2d(64),
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # Cq.Conv2d_quadratic(64, 192, kernel_size=3, padding=1),
            # nn.BatchNorm2d(192),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # Cq.Conv2d_quadratic(192, 384, kernel_size=3, padding=1),
            # nn.BatchNorm2d(384),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Cq.Conv2d_quadratic(384, 256, kernel_size=3, padding=1),
            # nn.BatchNorm2d(256),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            Cq.Conv2d_quadratic(256, 256, kernel_size=3, padding=1),
            # nn.BatchNorm2d(256),
            # nn.Conv2d(256, 256, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.classifier(x)
        return x