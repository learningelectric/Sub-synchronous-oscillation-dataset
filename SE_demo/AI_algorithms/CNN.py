import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, input_channels=1, height=1000, width=8, output_units=10, conv_layers_config=None,
                 num_fc_layers=2, hidden_units=128, kernel_size=(3, 3), pool_size=(2, 2), nonlin=nn.ReLU,
                 dropout_rate=0.2, stride=1, padding=2):
        super(CNN, self).__init__()

        # 设置默认的卷积层配置
        if conv_layers_config is None:
            conv_layers_config = [16, 64, 16]

        self.conv_layers_config = conv_layers_config
        self.num_fc_layers = num_fc_layers

        # 定义卷积层
        self.conv_layers = nn.ModuleList()
        current_channels = input_channels
        for out_channels in conv_layers_config:
            self.conv_layers.append(nn.Conv2d(current_channels, out_channels, kernel_size, stride=stride, padding=padding))
            self.conv_layers.append(nn.BatchNorm2d(out_channels))
            self.conv_layers.append(nonlin())
            self.conv_layers.append(nn.MaxPool2d(pool_size))
            self.conv_layers.append(nn.Dropout(dropout_rate))
            current_channels = out_channels

        # 计算全连接层的输入特征数
        self.num_features = self._calculate_num_features(input_channels, height, width)

        # 定义全连接层
        self.fc_layers = nn.ModuleList()
        current_units = self.num_features
        for i in range(num_fc_layers - 1):
            self.fc_layers.append(nn.Linear(current_units, hidden_units))
            self.fc_layers.append(nonlin())
            self.fc_layers.append(nn.Dropout(dropout_rate))
            current_units = hidden_units
        self.fc_layers.append(nn.Linear(current_units, output_units))  # Fixed this line

    def _calculate_num_features(self, channels, height, width):
        # 测试通过一个虚拟的输入数据来计算全连接层的输入特征数
        with torch.no_grad():
            x = torch.zeros(1, channels, height, width)
            for layer in self.conv_layers:
                x = layer(x)
            return x.numel()

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        x = x.view(x.size(0), -1)
        for layer in self.fc_layers:
            x = layer(x)
        return x
