import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch
import os


def create_model(cfg):
    if cfg.model == "VGG":
        model = models.vgg16(pretrained=True)
        model.classifier[-1] = nn.Linear(in_features=4096, out_features=2)
    else:
        model = Net()
    return model


def save_model(cfg, model, epoch, save_dir, device):
    save_filename = "%s_net_%s.pth" % (cfg.name, epoch)
    save_path = os.path.join(save_dir, save_filename).replace(
        "\\", "/"
    )

    if torch.cuda.is_available():
        torch.save(
            {
                "net": model.module.cpu().state_dict(),
            },
            save_path,
        )
        net.to(device)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
