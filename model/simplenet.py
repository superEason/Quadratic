import torch.nn as nn
import model.Conv2d_quadratic as Cq
import model.Linear_quadratic as Lq

'''
modified to fit dataset size
'''
NUM_CLASSES = 10

# ONLY Linear layer
class SimpleNet_0(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(SimpleNet_0, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(28*28, num_classes),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# ONLY quadratic Linear layer
class SimpleNet_1(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(SimpleNet_1, self).__init__()
        self.classifier = nn.Sequential(
            Lq.Linear_quadratic(28*28, num_classes),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# One convolution layer
class SimpleNet_2(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(SimpleNet_2, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.ReLU (inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16*14*14, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Quadratic convolution layer
class SimpleNet_3(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(SimpleNet_3, self).__init__()
        self.features = nn.Sequential(
            Cq.Conv2d_quadratic(1, 16, kernel_size=5, padding=2),
            nn.ReLU (inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16*14*14, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Quadratic convolution layer
class SimpleNet_4(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(SimpleNet_4, self).__init__()
        self.features = nn.Sequential(
            Cq.Conv2d_quadratic(1, 16, kernel_size=5, padding=2),
            nn.ReLU (inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            Lq.Linear_quadratic(16*14*14, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

#Without ReLu
class SimpleNet_5(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(SimpleNet_5, self).__init__()
        self.features = nn.Sequential(
            Cq.Conv2d_quadratic(1, 16, kernel_size=5, padding=2),
            # nn.ReLU (inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16*14*14, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Quadratic convolution layer
class SimpleNet_6(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(SimpleNet_6, self).__init__()
        self.features = nn.Sequential(
            Cq.Conv2d_quadratic(1, 2, kernel_size=5, padding=2),
            nn.ReLU (inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            Lq.Linear_quadratic(2*14*14, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Quadratic convolution layer
class SimpleNet_7(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(SimpleNet_7, self).__init__()
        self.features = nn.Sequential(
            Cq.Conv2d_quadratic(1, 16, kernel_size=5, padding=2),
            nn.ReLU (inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            Lq.Linear_quadratic(16*14*14, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x