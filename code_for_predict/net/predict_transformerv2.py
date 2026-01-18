import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import pdb
import torch
import torch.nn as nn
from net.Transformer_encoder import Predict_encoder

class ResidualBlock(nn.Module):
    def __init__(self, num_channels,kernel_size,padding):
        super(ResidualBlock, self).__init__()
        self.num_channels = num_channels

        # 定义两个卷积层
        self.conv1 = nn.Conv1d(num_channels, num_channels, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(num_channels, num_channels, kernel_size=kernel_size, padding=padding)
        self.batch_norm = nn.BatchNorm1d(num_channels) 

        self.ac = nn.LeakyReLU()
        
    def forward(self, x):
        res = x
        for _ in range(2):
            res = self.conv1(res)
            res = self.batch_norm(res) 
            # res = F.relu(res)
            res = self.ac(res)

            res = self.conv2(res)
            res = self.batch_norm(res)
            # res = F.relu(res)
            
        return x + res


class Predict_translation_on_off(torch.nn.Module):
    
    def __init__(self,params):
        super(Predict_translation_on_off, self).__init__()

        self.dropout_rate_fc = params['dropout_rate_fc']
        self.relu = nn.ReLU()

        self.trans_ori_pos = Predict_encoder(nhead = params['num_head1'],layers = params['transformer_num_layers1'],hidden_dim=params['hidden_dim1'],latent_dim=params['latent_dim1'],embedding_dim=params['embedding_dim1'],seq_len=params['seq_len'],probs=params['dropout_rate1'],device='cuda')
        self.trans_dim_pos = Predict_encoder(nhead = params['num_head2'],layers = params['transformer_num_layers2'],hidden_dim=params['hidden_dim2'],latent_dim=params['latent_dim2'],embedding_dim=params['embedding_dim2'],seq_len=params['seq_len'],probs=params['dropout_rate2'],device='cuda')
        # self.trans_all = Predict_encoder(nhead = 4,layers=4,hidden_dim=4,latent_dim=64,embedding_dim=100,seq_len=100,probs=0.1,device='cuda')
        
        # Define the layers as PyTorch modules
        self.embedding_ori = torch.nn.Embedding(100, params['embedding_dim1'])
        self.embedding_dim = torch.nn.Embedding(100, params['embedding_dim2'])
        # self.embedding_pos = torch.nn.Embedding(100, params['embedding_dim'])
        
        # Define 1D-CNN
        # self.cnn_ori = ResidualBlock(num_channels=100,kernel_size=2*params['conv1d_padding']+1,padding=params['conv1d_padding'])
        # self.cnn_dim = ResidualBlock(num_channels=100,kernel_size=2*params['conv1d_padding']+1,padding=params['conv1d_padding'])
        # self.cnn_all = ResidualBlock(num_channels=100,kernel_size=2*params['conv1d_padding']+1,padding=params['conv1d_padding'])
        # self.cnn_ori_dim = ResidualBlock(num_channels=100,kernel_size=2*params['conv1d_padding']+1,padding=params['conv1d_padding'])

        
        # dropout层
        self.ac = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=self.dropout_rate_fc)
        
        self.final_fc1 = nn.Linear(params['latent_dim1'] + params['latent_dim2'], params['fc_hidden1']) # 其中3+3+2+1=9为生物信息的预留位置
        self.final_fc2 = nn.Linear(params['fc_hidden1'],params['fc_hidden2'])
        self.final_fc3 = nn.Linear(params['fc_hidden2'],1)
        # self.bio_fc1 = nn.Linear(13, params['fc_hidden1'])
        
    def forward(self, X):
        # print('X_in.ori = ', X[1,0,0,:])
        # print('X_in.dim = ', X[1,0,1,:])
        # print('X_in.pos = ', X[1,0,2,:])
        x = X.to(torch.int)
        # bio = bio.to(torch.float)
        # Split input X: [bt,1,3,100]
        # x = X[:,0,:,:]
        # print('x = ',x[1])
        input_ori = x[:, 0, :]
        input_dim = x[:, 1, :]
        # input_pos = x[:, 2, :] - 1
        # print('input_ori.shape = ',input_ori.shape)
        # print('input_dim.shape = ',input_dim.shape)
        # print('input_pos.shape = ',input_pos.shape)
        # print('input_ori = ',input_ori)
        
        # print('start embedding')
        embeded_ori = self.embedding_ori(input_ori)
        embeded_dim = self.embedding_dim(input_dim)
        # embeded_pos = self.embedding_pos(input_pos)
        
        # 进入卷积模块
        
        # embeded_dim = self.cnn_dim(embeded_dim)
        # embeded_ori = self.cnn_ori(embeded_ori)
        # embeded_ori_dim = self.cnn_ori_dim( embeded_ori +  embeded_dim )
        
        # cnn_all = self.cnn_all(embeded_ori_dim + embeded_pos)
      
        # print('embeded_ori.shape = ', embeded_ori.shape)
        
        # print('start transformer encoder')
        # all_trans = self.trans_all(embeded_ori_dim)
        
        ori_pos = self.trans_ori_pos(embeded_ori)
        dim_pos = self.trans_dim_pos(embeded_dim)
        # print('end transformer encoder')
        
        output = torch.cat((ori_pos, dim_pos), dim=-1) # 将transformer的输出和生物信息相融合
        # output = self.mlp(ori_dim_pos)
        
        output = self.final_fc1(output)
        output = self.ac(output)
        # output = self.relu(output)
        output = self.dropout(output)

        output = self.final_fc2(output)
        output = self.ac(output)


        output = self.final_fc3(output)

        return self.relu(output)



class Predict_translation_off(torch.nn.Module):
    
    def __init__(self,params):
        super(Predict_translation_off, self).__init__()

        self.dropout_rate_fc = params['dropout_rate_fc']
        self.relu = nn.ReLU()

        self.trans_ori_pos = Predict_encoder(nhead = params['num_head1'],layers = params['transformer_num_layers1'],hidden_dim=params['hidden_dim1'],latent_dim=params['latent_dim1'],embedding_dim=params['embedding_dim1'],seq_len=params['seq_len'],probs=params['dropout_rate1'],device='cuda')
        self.trans_dim_pos = Predict_encoder(nhead = params['num_head2'],layers = params['transformer_num_layers2'],hidden_dim=params['hidden_dim2'],latent_dim=params['latent_dim2'],embedding_dim=params['embedding_dim2'],seq_len=params['seq_len'],probs=params['dropout_rate2'],device='cuda')

        self.embedding_ori = torch.nn.Embedding(100, params['embedding_dim1'])
        self.embedding_dim = torch.nn.Embedding(100, params['embedding_dim2'])

        self.ac = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=self.dropout_rate_fc)
        
        self.final_fc1 = nn.Linear(params['latent_dim1'] + params['latent_dim2'], params['fc_hidden1']) # 其中3+3+2+1=9为生物信息的预留位置
        self.final_fc2 = nn.Linear(params['fc_hidden1'],params['fc_hidden2'])
        self.final_fc3 = nn.Linear(params['fc_hidden2'],1)

        
    def forward(self, X):

        x = X.to(torch.int)
 
        input_ori = x[:, 0, :]
        input_dim = x[:, 1, :]
        # input_pos = x[:, 2, :] - 1
        # print('input_ori.shape = ',input_ori.shape)
        # print('input_dim.shape = ',input_dim.shape)
        # print('input_pos.shape = ',input_pos.shape)
        # print('input_ori = ',input_ori)
        
        # print('start embedding')
        embeded_ori = self.embedding_ori(input_ori)
        embeded_dim = self.embedding_dim(input_dim)
        # embeded_pos = self.embedding_pos(input_pos)
        
        # 进入卷积模块
        
        # embeded_dim = self.cnn_dim(embeded_dim)
        # embeded_ori = self.cnn_ori(embeded_ori)
        # embeded_ori_dim = self.cnn_ori_dim( embeded_ori +  embeded_dim )
        
        # cnn_all = self.cnn_all(embeded_ori_dim + embeded_pos)
      
        # print('embeded_ori.shape = ', embeded_ori.shape)
        
        # print('start transformer encoder')
        # all_trans = self.trans_all(embeded_ori_dim)
        
        ori_pos = self.trans_ori_pos(embeded_ori)
        dim_pos = self.trans_dim_pos(embeded_dim)
        # print('end transformer encoder')
        
        output = torch.cat((ori_pos, dim_pos), dim=-1) # 将transformer的输出和生物信息相融合
        # output = self.mlp(ori_dim_pos)
        
        output = self.final_fc1(output)
        output = self.ac(output)
        # output = self.relu(output)
        output = self.dropout(output)

        output = self.final_fc2(output)
        output = self.ac(output)
        # output = self.dropout(output)

        output = self.final_fc3(output)
        
        # output = self.relu(output)
        # print('output.shape', output.shape)
        # pdb.set_trace()
        return self.relu(output)


class Predict_translation(torch.nn.Module):
    
    def __init__(self,params):
        super(Predict_translation, self).__init__()

        self.dropout_rate_fc = params['dropout_rate_fc']
        self.relu = nn.ReLU()

        self.trans_ori_pos = Predict_encoder(nhead = params['num_head1'],layers = params['transformer_num_layers1'],hidden_dim=params['hidden_dim1'],latent_dim=params['latent_dim1'],embedding_dim=params['embedding_dim1'],seq_len=params['seq_len'],probs=params['dropout_rate1'],device='cuda')
        self.trans_dim_pos = Predict_encoder(nhead = params['num_head2'],layers = params['transformer_num_layers2'],hidden_dim=params['hidden_dim2'],latent_dim=params['latent_dim2'],embedding_dim=params['embedding_dim2'],seq_len=params['seq_len'],probs=params['dropout_rate2'],device='cuda')

        self.embedding_ori = torch.nn.Embedding(100, params['embedding_dim1'])
        self.embedding_dim = torch.nn.Embedding(100, params['embedding_dim2'])
        
        # dropout层
        self.ac = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=self.dropout_rate_fc)
        
        self.final_fc1 = nn.Linear(params['latent_dim1'] + params['latent_dim2'], params['fc_hidden1']) # 其中3+3+2+1=9为生物信息的预留位置
        self.final_fc2 = nn.Linear(params['fc_hidden1'],params['fc_hidden2'])
        self.final_fc3 = nn.Linear(params['fc_hidden2'],1)
        # self.bio_fc1 = nn.Linear(13, params['fc_hidden1'])
        
    def forward(self, X):

        x = X.to(torch.int)

        input_ori = x[:, 0, :]
        input_dim = x[:, 1, :]

        embeded_ori = self.embedding_ori(input_ori)
        embeded_dim = self.embedding_dim(input_dim)
        
        ori_pos = self.trans_ori_pos(embeded_ori)
        dim_pos = self.trans_dim_pos(embeded_dim)

        
        output = torch.cat((ori_pos, dim_pos), dim=-1) # 将transformer的输出和生物信息相融合
        
        output = self.final_fc1(output)
        output = self.ac(output)

        output = self.dropout(output)

        output = self.final_fc2(output)
        output = self.ac(output)
 

        output = self.final_fc3(output)

        return self.relu(output)





class CNN1D_Flatten(nn.Module):
    
    def __init__(self, fc_hidden=512):
        super(CNN1D_Flatten, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=115, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.pool1 = nn.MaxPool1d(kernel_size=5, stride=5)  # 输出 23

        self.conv2 = nn.Conv1d(128, 256, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool1d(kernel_size=5, stride=5)  # 输出 4

        self.conv3 = nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1)  # 输出仍为 4

        # 512 × 4 = 2048
        self.fc = nn.Linear(2048, fc_hidden)
        
    def forward(self, x):
        
        # print(x.shape)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        # print(x.shape)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        # print(x.shape)
        x = F.relu(self.conv3(x))  # 输出形状 [B, 256, L]
        # print(x.shape)
        x = x.flatten(start_dim=1)  # 展平为 [B, 256 * L]
        # print(x.shape)
        x = self.fc(x)              # 映射为 [B, fc_hidden]

        return x


class CrossAttentionFusionSameDim(nn.Module):
    
    def __init__(self, dim=256, num_heads=4, dropout=0.1):
        super().__init__()

        self.attn1 = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.attn2 = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.attn3 = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)

        self.fusion = nn.Linear(dim * 3, dim)  # 三个 attention 输出融合

    def forward(self, x1, x2, x3):
        # 输入：[B, D] → [B, 1, D]
        x1 = x1.unsqueeze(1)
        x2 = x2.unsqueeze(1)
        x3 = x3.unsqueeze(1)

        # Cross Attention：每个向量 attends 另一个
        o1, _ = self.attn1(query=x1, key=x2, value=x2)  # x1 attends x2
        o2, _ = self.attn2(query=x2, key=x3, value=x3)  # x2 attends x3
        o3, _ = self.attn3(query=x3, key=x1, value=x1)  # x3 attends x1

        # 拼接输出 + 降维
        fused = torch.cat([o1, o2, o3], dim=-1)  # [B, 1, 3*D]
        fused = self.fusion(fused)              # [B, 1, D]
        return fused.squeeze(1)                 # [B, D]
#从这里开始修改#
# 四分支交叉注意力融合模块
# ============================================================
class CrossAttentionFusionFourBranches(nn.Module):
    """
    融合四个特征分支（Ernie + Ori + Dim + Structure）
    """
    
    def __init__(self, dim=256, num_heads=4, dropout=0.1):
        super().__init__()
        
        # 四个分支之间的交叉注意力
        self.attn12 = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.attn23 = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.attn34 = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.attn41 = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        
        # 融合层
        self.fusion = nn.Linear(dim * 4, dim)
        self.layer_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x1, x2, x3, x4):
        """
        Args:
            x1: (batch, latent_dim) - Ernie 特征
            x2: (batch, latent_dim) - Ori 特征
            x3: (batch, latent_dim) - Dim 特征
            x4: (batch, latent_dim) - Structure 特征
        """
        # 扩展维度以适配 MultiheadAttention
        x1 = x1.unsqueeze(1)  # (batch, 1, dim)
        x2 = x2.unsqueeze(1)
        x3 = x3.unsqueeze(1)
        x4 = x4.unsqueeze(1)

        # 交叉注意力
        o1, _ = self.attn12(query=x1, key=x2, value=x2)  # x1 attends x2
        o2, _ = self.attn23(query=x2, key=x3, value=x3)  # x2 attends x3
        o3, _ = self.attn34(query=x3, key=x4, value=x4)  # x3 attends x4
        o4, _ = self.attn41(query=x4, key=x1, value=x1)  # x4 attends x1

        # 拼接所有输出
        fused = torch.cat([o1, o2, o3, o4], dim=-1)  # (batch, 1, 4*dim)
        
        # 降维融合
        fused = self.fusion(fused)  # (batch, 1, dim)
        fused = self.layer_norm(fused)
        fused = self.dropout(fused)
        
        return fused.squeeze(1)  # (batch, dim)
    
class CnnFcFeatureExtractor(nn.Module):
    """
    CNN 串联全连接神经网络，用于处理 (Batch_Size, Sequence_Length, Feature_Dim) 的特征，
    并映射到 params['latent_dim']。

    假设输入形状为 (Batch_Size, Sequence_Length, Feature_Dim)。
    例如，如果输入是 (258, 115, 768)，则 Batch_Size=258, Sequence_Length=115, Feature_Dim=768。
    """
    def __init__(self, params):
        super(CnnFcFeatureExtractor, self).__init__()

        # 从 params 中获取模型结构参数
        self.input_sequence_length = params['seq_len'] # 使用 params['seq_len']
        self.input_feature_dim = params['ernie_embedding_dim'] # 使用 params['ernie_embedding_dim'] (768)
        self.latent_dim = params['latent_dim']             # 目标输出维度

        self.cnn_channels_1 = params.get('cnn_ernie_channels_1', 256)
        self.cnn_kernel_size_1 = params.get('cnn_ernie_kernel_size_1', 7)
        self.cnn_padding_1 = params.get('cnn_ernie_padding_1', 3)
        self.pool_kernel_size_1 = params.get('pool_ernie_kernel_size_1', 3)
        self.pool_stride_1 = params.get('pool_ernie_stride_1', 2)

        self.cnn_channels_2 = params.get('cnn_ernie_channels_2', 512)
        self.cnn_kernel_size_2 = params.get('cnn_ernie_kernel_size_2', 5)
        self.cnn_padding_2 = params.get('cnn_ernie_padding_2', 2)
        self.pool_kernel_size_2 = params.get('pool_ernie_kernel_size_2', 3)
        self.pool_stride_2 = params.get('pool_ernie_stride_2', 2)

        self.cnn_channels_3 = params.get('cnn_ernie_channels_3', 1024)
        self.cnn_kernel_size_3 = params.get('cnn_ernie_kernel_size_3', 3)
        self.cnn_padding_3 = params.get('cnn_ernie_padding_3', 1)

        self.dropout_rate = params.get('dropout_rate_cnn_fc', params['dropout_rate_fc']) # 可以共享或独立

        # ==================== CNN Layers ====================
        # 输入形状: (Batch_Size, Feature_Dim, Sequence_Length)
        self.conv1 = nn.Conv1d(
            in_channels=self.input_feature_dim,
            out_channels=self.cnn_channels_1,
            kernel_size=self.cnn_kernel_size_1,
            padding=self.cnn_padding_1
        )
        self.bn1 = nn.BatchNorm1d(self.cnn_channels_1)
        self.pool1 = nn.MaxPool1d(
            kernel_size=self.pool_kernel_size_1,
            stride=self.pool_stride_1
        )

        self.conv2 = nn.Conv1d(
            in_channels=self.cnn_channels_1,
            out_channels=self.cnn_channels_2,
            kernel_size=self.cnn_kernel_size_2,
            padding=self.cnn_padding_2
        )
        self.bn2 = nn.BatchNorm1d(self.cnn_channels_2)
        self.pool2 = nn.MaxPool1d(
            kernel_size=self.pool_kernel_size_2,
            stride=self.pool_stride_2
        )

        self.conv3 = nn.Conv1d(
            in_channels=self.cnn_channels_2,
            out_channels=self.cnn_channels_3,
            kernel_size=self.cnn_kernel_size_3,
            padding=self.cnn_padding_3
        )
        self.bn3 = nn.BatchNorm1d(self.cnn_channels_3)
        self.dropout_cnn = nn.Dropout(self.dropout_rate)


        # ==================== 计算展平后的维度 ====================
        self._calculate_flattened_dim()


        # ==================== FC Layers ====================
        self.fc1 = nn.Linear(self.flattened_dim, self.latent_dim * 2) # 中间层可以更大一些
        self.bn_fc1 = nn.BatchNorm1d(self.latent_dim * 2)
        self.fc2 = nn.Linear(self.latent_dim * 2, self.latent_dim)
        self.dropout_fc = nn.Dropout(self.dropout_rate)

    def _calculate_flattened_dim(self):
        """
        计算 CNN 输出展平后的维度，用于初始化全连接层。
        """
        # 创建一个假输入来模拟前向传播
        # Batch_Size=1, Feature_Dim=self.input_feature_dim, Sequence_Length=self.input_sequence_length
        dummy_input = torch.zeros(1, self.input_feature_dim, self.input_sequence_length) # 修正这里

        x = F.relu(self.bn1(self.conv1(dummy_input)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        
        self.flattened_dim = x.numel() # 获取展平后的元素总数
        print(f"CNN (Ernie branch) Output Flattened Dimension: {self.flattened_dim}")
        
    def forward(self, x):
        """
        前向传播函数。
        Args:
            x: 输入特征张量，形状为 (Batch_Size, Sequence_Length, Feature_Dim)。
               例如 (258, 115, 768)。
        Returns:
            映射到 params['latent_dim'] 维度的特征张量，形状为 (Batch_Size, latent_dim)。
        """
        # 将输入形状从 (Batch_Size, Sequence_Length, Feature_Dim)
        # 转换为 (Batch_Size, Feature_Dim, Sequence_Length) 以适应 Conv1d
        x = x.transpose(1, 2) # 变为 (Batch_Size, 768, 115)

        # ==================== CNN Layers ====================
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x) 
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x) 
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout_cnn(x)

        # ==================== Flatten ====================
        x = x.flatten(start_dim=1) 

        # ==================== FC Layers ====================
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout_fc(x)
        x = self.fc2(x) # 输出形状: (Batch_Size, latent_dim)

        return x
    
class Predict_translation_structure(torch.nn.Module): # sequence expert + dimer expert + 二级结构
    
    def __init__(self,params):
        super(Predict_translation_structure, self).__init__()

        self.dropout_rate_fc = params['dropout_rate_fc']
        self.relu = nn.ReLU()

        # self.ernie_projection = nn.Linear(768, params['embedding_dim1'])
        # self.ernie_layer_norm = nn.LayerNorm(params['latent_dim'])

            
        # self.trans_ernie = Predict_encoder(
        #     nhead=params['num_head1'],
        #     layers=params['transformer_num_layers1'],
        #     hidden_dim=params['hidden_dim1'],
        #     latent_dim=params['latent_dim'],
        #     embedding_dim=params['latent_dim'],
        #     seq_len=params['seq_len'],
        #     probs=params['dropout_rate1'],
        #     device='cuda'
        # )
        # ==================== 新增 Ernie CNN-FC 特征提取器 ====================
        # 为 CnnFcFeatureExtractor 准备参数
        # ============ Ernie CnnFcFeatureExtractor 部分 ============
        ernie_cnn_fc_params = {
            'seq_len': params['seq_len'],  # 例如 115
            'ernie_embedding_dim': params['embedding_dim1'],  # RNA-Ernie的嵌入维度
            'latent_dim': params['latent_dim'],
            
            # CNN-FC 特定参数
            'cnn_ernie_channels_1': params.get('cnn_ernie_channels_1', 256),
            'cnn_ernie_kernel_size_1': params.get('cnn_ernie_kernel_size_1', 7),
            'cnn_ernie_padding_1': params.get('cnn_ernie_padding_1', 3),
            'pool_ernie_kernel_size_1': params.get('pool_ernie_kernel_size_1', 3),
            'pool_ernie_stride_1': params.get('pool_ernie_stride_1', 2),

            'cnn_ernie_channels_2': params.get('cnn_ernie_channels_2', 512),
            'cnn_ernie_kernel_size_2': params.get('cnn_ernie_kernel_size_2', 5),
            'cnn_ernie_padding_2': params.get('cnn_ernie_padding_2', 2),
            'pool_ernie_kernel_size_2': params.get('pool_ernie_kernel_size_2', 3),
            'pool_ernie_stride_2': params.get('pool_ernie_stride_2', 2),

            'cnn_ernie_channels_3': params.get('cnn_ernie_channels_3', 1024),
            'cnn_ernie_kernel_size_3': params.get('cnn_ernie_kernel_size_3', 3),
            'cnn_ernie_padding_3': params.get('cnn_ernie_padding_3', 1),
            
            'dropout_rate_cnn_fc': params.get('dropout_rate_cnn_fc', self.dropout_rate_fc),
            'dropout_rate_fc': params['dropout_rate_fc'],
        }
        self.ernie_cnn_fc_extractor = CnnFcFeatureExtractor(ernie_cnn_fc_params)


        self.trans_ori_pos = Predict_encoder(nhead = params['num_head1'],layers = params['transformer_num_layers1'],hidden_dim=params['hidden_dim1'],latent_dim=params['latent_dim'],embedding_dim=params['embedding_dim1'],seq_len=params['seq_len'],probs=params['dropout_rate1'],device='cuda')
        self.trans_dim_pos = Predict_encoder(nhead = params['num_head2'],layers = params['transformer_num_layers2'],hidden_dim=params['hidden_dim2'],latent_dim=params['latent_dim'],embedding_dim=params['embedding_dim2'],seq_len=params['seq_len'],probs=params['dropout_rate2'],device='cuda')

        self.embedding_ori = torch.nn.Embedding(100, params['embedding_dim1'])
        self.embedding_dim = torch.nn.Embedding(100, params['embedding_dim2'])
        self.cnn = CNN1D_Flatten(fc_hidden=params['latent_dim'])#结构信息分支
        #四分支交叉注意力融合
        self.CrossAttention = CrossAttentionFusionFourBranches(
            dim=params['latent_dim'],
            num_heads=4,
            dropout=self.dropout_rate_fc
        )
        # dropout层
        self.ac = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=self.dropout_rate_fc)
        
        self.final_fc1 = nn.Linear(params['latent_dim'], params['fc_hidden1']) # 其中3+3+2+1=9为生物信息的预留位置
        self.final_fc2 = nn.Linear(params['fc_hidden1'],params['fc_hidden2'])
        self.final_fc3 = nn.Linear(params['fc_hidden2'],1)
 
        
    def forward(self, X, structure, ernie_embedding):

        # ============ RNA-Ernie 分支处理 ============
        # print("data",X.shape)
        # print("struc",structure.shape)
        # print("ernie",ernie_embedding.shape)
        # ernie_features = self.ernie_projection(ernie_embedding)
        # ernie_features = self.ernie_layer_norm(ernie_features)
        ernie_output = self.ernie_cnn_fc_extractor(ernie_embedding)  # (batch, latent_dim)



        x = X.to(torch.int)
        
        structure = self.cnn(structure.to(torch.float))

        input_ori = x[:, 0, :]
        input_dim = x[:, 1, :]

        embeded_ori = self.embedding_ori(input_ori)
        embeded_dim = self.embedding_dim(input_dim)
        
        ori_pos = self.trans_ori_pos(embeded_ori)
        dim_pos = self.trans_dim_pos(embeded_dim)

        
        # output = torch.cat((ori_pos, dim_pos), dim=-1) # 将transformer的输出和生物信息相融合
        
        output = self.CrossAttention(ernie_output, dim_pos, ori_pos, structure)
        
        output = self.final_fc1(output)
        output = self.ac(output)

        output = self.dropout(output)

        output = self.final_fc2(output)
        output = self.ac(output)
 

        output = self.final_fc3(output)

        return self.relu(output)
    
    
    
class CNN_OneHot_Seq(nn.Module):
    def __init__(self, input_length=115, task_type='regression'):
        super(CNN_OneHot_Seq, self).__init__()

        # 输入通道为4（A、C、G、U 的 one-hot 编码）
        self.conv1 = nn.Conv1d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.3)

        # 展平后尺寸：128 × input_length
        self.flattened_dim = 128 * input_length
        self.fc1 = nn.Linear(self.flattened_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.bn_fc1 = nn.BatchNorm1d(16)
        self.bn_fc2 = nn.BatchNorm1d(16)

        if task_type == 'regression':
            self.out = nn.Linear(16, 1)  # ON, OFF, ON/OFF
        elif task_type == 'classification':
            self.out = nn.Linear(16, 2)  # softmax 输出
        else:
            raise ValueError("task_type must be 'regression' or 'classification'")

        self.task_type = task_type
        

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)

        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout(x)

        x = self.out(x)

        if self.task_type == 'classification':
            x = F.softmax(x, dim=1)

        return x






if __name__ == "__main__":
    
    cnn = CNN1D_Flatten(fc_hidden=128)
    input_tensor = torch.randn(64, 115, 115)
    output_tensor = cnn(input_tensor)