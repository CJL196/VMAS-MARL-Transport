import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, num_layers=3):
        super(MLP, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.Tanh())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)

class Actor(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=128):
        super(Actor, self).__init__()
        self.net = MLP(input_dim, hidden_dim, hidden_dim) 
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        # Log STD 作为可训练参数
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        # Output Action weights init
        nn.init.orthogonal_(self.mean_layer.weight, 0.01)
        nn.init.constant_(self.mean_layer.bias, 0)

    def forward(self, x):
        features = self.net(x)
        mean = self.mean_layer(features)
        # 限制 Log STD 范围，防止数值不稳定
        log_std = torch.clamp(self.log_std, -20, 2)
        std = torch.exp(log_std)
        return mean, std

class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(Critic, self).__init__()
        self.net = MLP(input_dim, 1, hidden_dim)
        
        # Value head weights init
        # Accessed via the last layer of MLP
        last_layer = self.net.net[-1]
        nn.init.orthogonal_(last_layer.weight, 1.0)
        nn.init.constant_(last_layer.bias, 0)

    def forward(self, x):
        return self.net(x)


