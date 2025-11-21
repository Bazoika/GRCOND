import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool, GCNConv, GINConv
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.utils import to_dense_adj, dense_to_sparse
import time

class GraphEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers=3):
        super(GraphEncoder, self).__init__()
        self.num_layers = num_layers
        
        # 使用GIN卷积层
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # 第一层
        nn1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.convs.append(GINConv(nn1))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # 中间层
        for _ in range(num_layers - 2):
            nn_mid = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINConv(nn_mid))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # 最后一层到潜在空间
        nn_final = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)  # 输出均值和方差
        )
        self.convs.append(GINConv(nn_final))
        
        self.pool = global_mean_pool
        
    def forward(self, x, edge_index, batch):
        # 图编码
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=0.1, training=self.training)
        
        # 最后一层，输出潜在分布的参数
        x = self.convs[-1](x, edge_index)
        
        # 全局池化得到图级表示
        graph_embedding = self.pool(x, batch)
        
        # 分割为均值和方差
        mu, log_var = torch.chunk(graph_embedding, 2, dim=-1)
        
        return mu, log_var

class GraphDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, max_nodes=30, feature_dim=37):
        super(GraphDecoder, self).__init__()
        self.max_nodes = max_nodes
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        
        # 从潜在变量生成初始节点特征
        self.node_generator = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_nodes * feature_dim)
        )
        
        # 边预测网络
        self.edge_predictor = nn.Sequential(
            nn.Linear(feature_dim * 2 + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, z):
        batch_size = z.size(0)
        
        # 生成节点特征
        node_features_flat = self.node_generator(z)
        node_features = node_features_flat.view(batch_size * self.max_nodes, self.feature_dim)
        
        # 为每个图创建所有可能的边
        edge_probs = []
        edge_indices = []
        
        for b in range(batch_size):
            # 获取当前图的节点特征
            start_idx = b * self.max_nodes
            end_idx = (b + 1) * self.max_nodes
            curr_nodes = node_features[start_idx:end_idx]
            
            # 创建所有可能的边组合
            src_nodes = []
            dst_nodes = []
            for i in range(self.max_nodes):
                for j in range(self.max_nodes):
                    if i != j:  # 避免自环
                        src_nodes.append(i)
                        dst_nodes.append(j)
            
            if len(src_nodes) == 0:
                edge_probs.append(None)
                edge_indices.append(None)
                continue
                
            # 准备边特征
            edge_src = curr_nodes[src_nodes]
            edge_dst = curr_nodes[dst_nodes]
            z_expanded = z[b:b+1].repeat(len(src_nodes), 1)
            
            edge_features = torch.cat([edge_src, edge_dst, z_expanded], dim=-1)
            
            # 预测边概率
            edge_prob = torch.sigmoid(self.edge_predictor(edge_features)).squeeze()
            edge_probs.append(edge_prob)
            
            # 调整边索引为全局索引
            global_src = torch.tensor(src_nodes, device=z.device) + start_idx
            global_dst = torch.tensor(dst_nodes, device=z.device) + start_idx
            edge_indices.append(torch.stack([global_src, global_dst], dim=0))
        
        return node_features, edge_probs, edge_indices

class GraphVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, latent_dim=64, max_nodes=30, num_encoder_layers=3):
        super(GraphVAE, self).__init__()
        self.encoder = GraphEncoder(input_dim, hidden_dim, latent_dim, num_encoder_layers)
        self.decoder = GraphDecoder(latent_dim, hidden_dim, max_nodes, input_dim)
        self.latent_dim = latent_dim
        
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x, edge_index, batch):
        # 编码
        mu, log_var = self.encoder(x, edge_index, batch)
        
        # 重参数化
        z = self.reparameterize(mu, log_var)
        
        # 解码
        reconstructed_nodes, edge_probs, edge_indices = self.decoder(z)
        
        return reconstructed_nodes, edge_probs, edge_indices, mu, log_var

def graph_reconstruction_loss(original_data, reconstructed_nodes, edge_probs, edge_indices, mu, log_var, beta=0.1):
    """
    计算图重建损失
    """
    batch_size = original_data.num_graphs
    max_nodes = reconstructed_nodes.size(0) // batch_size
    total_loss = 0
    
    # KL散度损失
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / batch_size
    
    for b in range(batch_size):
        # 获取原始图数据
        graph_mask = original_data.batch == b
        original_nodes = original_data.x[graph_mask]
        original_edge_index = original_data.edge_index[:, original_data.batch[original_data.edge_index[0]] == b]
        
        # 调整原始边索引为当前图内的局部索引
        node_start = torch.where(graph_mask)[0][0]
        local_edge_index = original_edge_index - node_start
        
        # 节点特征重建损失
        start_idx = b * max_nodes
        end_idx = (b + 1) * max_nodes
        recon_nodes = reconstructed_nodes[start_idx:end_idx]
        
        # 确保形状匹配 - 只比较实际存在的节点
        original_num_nodes = original_nodes.size(0)
        min_nodes = min(original_num_nodes, max_nodes)
        
        if min_nodes > 0:
            node_loss = F.mse_loss(
                recon_nodes[:min_nodes], 
                original_nodes[:min_nodes]
            )
        else:
            node_loss = torch.tensor(0.0, device=original_nodes.device)
        
        # 边重建损失
        edge_loss = torch.tensor(0.0, device=original_nodes.device)
        if len(edge_probs) > b and edge_probs[b] is not None and len(edge_probs[b]) > 0:
            try:
                # 创建真实的邻接矩阵 - 只考虑有效节点
                adj_real = torch.zeros(max_nodes, max_nodes, device=original_nodes.device)
                
                # 过滤掉超出范围的边索引
                valid_edge_mask = (local_edge_index[0] < max_nodes) & (local_edge_index[1] < max_nodes)
                valid_local_edge_index = local_edge_index[:, valid_edge_mask]
                
                if valid_local_edge_index.size(1) > 0:
                    adj_real[valid_local_edge_index[0], valid_local_edge_index[1]] = 1.0
                
                # 创建预测的邻接矩阵
                adj_pred = torch.zeros(max_nodes, max_nodes, device=original_nodes.device)
                if len(edge_indices) > b and edge_indices[b] is not None:
                    # 将边索引转换为局部索引
                    local_src = edge_indices[b][0] - start_idx
                    local_dst = edge_indices[b][1] - start_idx
                    
                    # 过滤掉超出范围的预测边
                    valid_pred_mask = (local_src < max_nodes) & (local_dst < max_nodes)
                    valid_local_src = local_src[valid_pred_mask]
                    valid_local_dst = local_dst[valid_pred_mask]
                    valid_edge_probs = edge_probs[b][valid_pred_mask]
                    
                    if valid_local_src.size(0) > 0:
                        adj_pred[valid_local_src, valid_local_dst] = valid_edge_probs
                
                # 边损失（只考虑有效的节点对）
                if min_nodes > 0:
                    edge_loss = F.binary_cross_entropy(
                        adj_pred[:min_nodes, :min_nodes],
                        adj_real[:min_nodes, :min_nodes]
                    )
            except Exception as e:
                print(f"边重建损失计算错误: {e}")
                edge_loss = torch.tensor(0.0, device=original_nodes.device)
        
        total_loss += node_loss + edge_loss
    
    total_loss = total_loss / batch_size + beta * kl_loss
    return total_loss
def train_graph_vae():
    # 加载NCI1数据集
    dataset_name = 'NCI1'

    dataset = TUDataset(
    '../data/'+dataset_name,
    dataset_name,
    
    #use_node_attr=True,
    )
    print(f'数据集: {dataset}')
    print(f'图数量: {len(dataset)}')
    print(f'节点特征维度: {dataset.num_features}')
    print(f'类别数: {dataset.num_classes}')
    
    # 数据加载器
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 模型参数
    input_dim = dataset.num_features
    hidden_dim = 128
    latent_dim = 64
    max_nodes = 30  # NCI1数据集中大部分图的节点数少于30
    
    # 初始化模型
    model = GraphVAE(input_dim, hidden_dim, latent_dim, max_nodes)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练循环
    model.train()
    losses = []
    start = time.time()
    for epoch in range(100):
        epoch_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            
            # 前向传播
            reconstructed_nodes, edge_probs, edge_indices, mu, log_var = model(
                batch.x, batch.edge_index, batch.batch
            )
            
            # 计算损失
            loss = graph_reconstruction_loss(
                batch, reconstructed_nodes, edge_probs, edge_indices, mu, log_var
            )
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {avg_loss:.4f}')
        print("epoch",time.time()-start)
    
    # 绘制损失曲线
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    
    return model

def generate_new_graphs(model, num_graphs=5):
    """生成新的图"""
    model.eval()
    
    with torch.no_grad():
        # 从标准正态分布采样
        z = torch.randn(num_graphs, model.latent_dim)
        
        # 解码生成新图
        node_features, edge_probs, edge_indices = model.decoder(z)
        
        generated_graphs = []
        
        for i in range(num_graphs):
            start_idx = i * model.decoder.max_nodes
            end_idx = (i + 1) * model.decoder.max_nodes
            
            nodes = node_features[start_idx:end_idx]
            
            if i < len(edge_indices) and edge_indices[i] is not None:
                # 应用阈值来获得二进制边
                threshold = 0.5
                edge_mask = edge_probs[i] > threshold
                edges = edge_indices[i][:, edge_mask]
                
                generated_graphs.append({
                    'node_features': nodes.cpu().numpy(),
                    'edge_index': edges.cpu().numpy(),
                    'edge_probs': edge_probs[i].cpu().numpy()
                })
            else:
                generated_graphs.append({
                    'node_features': nodes.cpu().numpy(),
                    'edge_index': np.array([]),
                    'edge_probs': np.array([])
                })
    
    return generated_graphs

# 训练模型
if __name__ == "__main__":
    # 训练图VAE模型
    trained_model = train_graph_vae()
    
    # 生成新图
    print("\n生成新图...")
    new_graphs = generate_new_graphs(trained_model, num_graphs=3)
    
    for i, graph in enumerate(new_graphs):
        print(f"\n图 {i+1}:")
        print(f"  节点数: {len(graph['node_features'])}")
        print(f"  边数: {graph['edge_index'].shape[1] if graph['edge_index'].size > 0 else 0}")
        print(f"  节点特征形状: {graph['node_features'].shape}")