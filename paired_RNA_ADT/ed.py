import torch
import torch.nn as nn

class NetBlock(nn.Module):
    def __init__(
            self,
            nlayer: int,
            dim_list: list,
            act_list: list,
            dropout_rate: float,
            noise_rate: float
    ):
        """
        multiple layers netblock with specific layer counts, dimension, activations and dropout.

        Parameters
        ----------
        nlayer
            layer counts.

        dim_list
            dimension list, length equal to nlayer + 1.

        act_list
            activation list, length equal to nlayer + 1.

        dropout_rate
            rate of dropout.

        noise_rate
            rate of set part of input data to 0.

        """
        super(NetBlock, self).__init__()
        self.nlayer = nlayer
        self.noise_dropout = nn.Dropout(noise_rate)
        self.linear_list = nn.ModuleList()
        self.bn_list = nn.ModuleList()
        self.activation_list = nn.ModuleList()
        self.dropout_list = nn.ModuleList()

        for i in range(nlayer):

            self.linear_list.append(nn.Linear(dim_list[i], dim_list[i + 1]))
            nn.init.xavier_uniform_(self.linear_list[i].weight)
            self.bn_list.append(nn.BatchNorm1d(dim_list[i + 1]))
            self.activation_list.append(act_list[i])
            if not i == nlayer - 1:
                self.dropout_list.append(nn.Dropout(dropout_rate))

    def forward(self, x):

        x = self.noise_dropout(x)
        for i in range(self.nlayer):
            x = self.linear_list[i](x)
            x = self.bn_list[i](x)
            x = self.activation_list[i](x)
            if not i == self.nlayer - 1:
                """ don't use dropout for output to avoid loss calculate break down """
                x = self.dropout_list[i](x)

        return x


