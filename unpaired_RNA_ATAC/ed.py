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



class Split_Chrom_Encoder_block(nn.Module):
    def __init__(
            self,
            nlayer: int,
            dim_list: list,
            act_list: list,
            chrom_list: list,
            dropout_rate: float,
            noise_rate: float
    ):
        """
        ATAC encoder netblock with specific layer counts, dimension, activations and dropout.

        Parameters
        ----------
        nlayer
            layer counts.

        dim_list
            dimension list, length equal to nlayer + 1.

        act_list
            activation list, length equal to nlayer + 1.

        chrom_list
            list record the peaks count for each chrom, assert that sum of chrom list equal to dim_list[0].

        dropout_rate
            rate of dropout.

        noise_rate
            rate of set part of input data to 0.

        """
        super(Split_Chrom_Encoder_block, self).__init__()
        self.nlayer = nlayer
        self.chrom_list = chrom_list
        self.noise_dropout = nn.Dropout(noise_rate)
        self.linear_list = nn.ModuleList()
        self.bn_list = nn.ModuleList()
        self.activation_list = nn.ModuleList()
        self.dropout_list = nn.ModuleList()

        for i in range(nlayer):
            if i == 0:
                """first layer seperately forward for each chrom"""
                self.linear_list.append(nn.ModuleList())
                self.bn_list.append(nn.ModuleList())
                self.activation_list.append(nn.ModuleList())
                self.dropout_list.append(nn.ModuleList())
                for j in range(len(chrom_list)):
                    self.linear_list[i].append(nn.Linear(chrom_list[j], dim_list[i + 1] // len(chrom_list)))
                    nn.init.xavier_uniform_(self.linear_list[i][j].weight)
                    self.bn_list[i].append(nn.BatchNorm1d(dim_list[i + 1] // len(chrom_list)))
                    self.activation_list[i].append(act_list[i])
                    self.dropout_list[i].append(nn.Dropout(dropout_rate))
            else:
                self.linear_list.append(nn.Linear(dim_list[i], dim_list[i + 1]))
                nn.init.xavier_uniform_(self.linear_list[i].weight)
                self.bn_list.append(nn.BatchNorm1d(dim_list[i + 1]))
                self.activation_list.append(act_list[i])
                if not i == nlayer - 1:
                    self.dropout_list.append(nn.Dropout(dropout_rate))

    def forward(self, x):

        x = self.noise_dropout(x)
        for i in range(self.nlayer):
            if i == 0:
                x = torch.split(x, self.chrom_list, dim=1)
                temp = []
                for j in range(len(self.chrom_list)):
                    temp.append(self.dropout_list[0][j](
                        self.activation_list[0][j](self.bn_list[0][j](self.linear_list[0][j](x[j])))))
                x = torch.concat(temp, dim=1)
            else:
                x = self.linear_list[i](x)
                x = self.bn_list[i](x)
                x = self.activation_list[i](x)
                if not i == self.nlayer - 1:
                    """ don't use dropout for output to avoid loss calculate break down """
                    x = self.dropout_list[i](x)
        return x


class Split_Chrom_Decoder_block(nn.Module):
    def __init__(
            self,
            nlayer: int,
            dim_list: list,
            act_list: list,
            chrom_list: list,
            dropout_rate: float,
            noise_rate: float
    ):
        """
        ATAC decoder netblock with specific layer counts, dimension, activations and dropout.

        Parameters
        ----------
        nlayer
            layer counts.

        dim_list
            dimension list, length equal to nlayer + 1.

        act_list
            activation list, length equal to nlayer + 1.

        chrom_list
            list record the peaks count for each chrom, assert that sum of chrom list equal to dim_list[end].

        dropout_rate
            rate of dropout.

        noise_rate
            rate of set part of input data to 0.

        """
        super(Split_Chrom_Decoder_block, self).__init__()
        self.nlayer = nlayer
        self.noise_dropout = nn.Dropout(noise_rate)
        self.chrom_list = chrom_list
        self.linear_list = nn.ModuleList()
        self.bn_list = nn.ModuleList()
        self.activation_list = nn.ModuleList()
        self.dropout_list = nn.ModuleList()

        for i in range(nlayer):
            if not i == nlayer - 1:
                self.linear_list.append(nn.Linear(dim_list[i], dim_list[i + 1]))
                nn.init.xavier_uniform_(self.linear_list[i].weight)
                self.bn_list.append(nn.BatchNorm1d(dim_list[i + 1]))
                self.activation_list.append(act_list[i])
                self.dropout_list.append(nn.Dropout(dropout_rate))
            else:
                """last layer seperately forward for each chrom"""
                self.linear_list.append(nn.ModuleList())
                self.bn_list.append(nn.ModuleList())
                self.activation_list.append(nn.ModuleList())
                self.dropout_list.append(nn.ModuleList())
                for j in range(len(chrom_list)):
                    self.linear_list[i].append(nn.Linear(dim_list[i] // len(chrom_list), chrom_list[j]))
                    nn.init.xavier_uniform_(self.linear_list[i][j].weight)
                    self.bn_list[i].append(nn.BatchNorm1d(chrom_list[j]))
                    self.activation_list[i].append(act_list[i])

    def forward(self, x):

        x = self.noise_dropout(x)
        for i in range(self.nlayer):
            if not i == self.nlayer - 1:
                x = self.linear_list[i](x)
                x = self.bn_list[i](x)
                x = self.activation_list[i](x)
                x = self.dropout_list[i](x)
            else:
                x = torch.chunk(x, len(self.chrom_list), dim=1)
                temp = []
                for j in range(len(self.chrom_list)):
                    temp.append(self.activation_list[i][j](self.bn_list[i][j](self.linear_list[i][j](x[j]))))
                x = torch.concat(temp, dim=1)

        return x




