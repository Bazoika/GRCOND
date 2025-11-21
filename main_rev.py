import os
import time
import copy
import argparse
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from torchvision.utils import save_image
from utils import get_loops, get_dataset, get_network, get_eval_pool, evaluate_synset, get_daparam, match_loss, get_time, TensorDataset, epoch, DiffAugment, ParamDiffAug
from sklearn.model_selection import train_test_split
from collections import defaultdict
import pandas as pd
import visdom
from torch import nn
from sklearn.manifold import TSNE
from torch.optim import Adam
from torch_geometric.loader import DataLoader

from torch_geometric.datasets import TUDataset,GNNBenchmarkDataset
from torch_geometric.data import Data
from torch.utils.data import Dataset
from tqdm import tqdm
import torch.nn.init as init
from torch_geometric.nn import global_add_pool
import random
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Batch
from utils import Indegree
from set_determ import set_determ
from networks import *
from sklearn import metrics
from sklearn.metrics import f1_score,jaccard_score




import warnings
import sys
sys.path.append("./generate_model") 
from autoencoder import VariationalAutoEncoder
sys.path.append("../gvae")
from GVAE import GVAE

warnings.filterwarnings('ignore')
def collate_fn(batch):
    return Batch.from_data_list(batch)
import networkx as nx
from itertools import combinations
import itertools
import networkx as nx


def average_cross_group_cosine_similarity(group_a, group_b):
    # 确保两组向量的维度相同
    if group_a.shape[1] != group_b.shape[1]:
        raise ValueError("两组向量的维度必须相同")  
    # 计算每组向量的范数
    group_a = group_a.cpu().detach().numpy()
    group_b = group_b.cpu().detach().numpy()
    norms_a = np.linalg.norm(group_a, axis=1)
    norms_b = np.linalg.norm(group_b, axis=1)
    
    # 避免除零错误
    norms_a[norms_a == 0] = 0.000001
    norms_b[norms_b == 0] = 0.000001
    
    # 计算所有向量对的余弦相似度
    similarity_matrix = np.dot(group_a, group_b.T) / np.outer(norms_a, norms_b)
    
    # 计算平均值
    total_pairs = similarity_matrix.size
    total_similarity = np.sum(similarity_matrix)
    avg_similarity = total_similarity / total_pairs
    
    return avg_similarity
def average_cosine_similarity(embeddings):

    n = embeddings.shape[0]  # 向量数量
    
    if n < 2:
        return 0.0
    embeddings = embeddings.cpu().detach().numpy()
    # 计算每个向量的范数 (L2范数)
    norms = np.linalg.norm(embeddings, axis=1)
    
    # 避免除零错误 (将零范数替换为1)
    norms[norms == 0] = 1
    
    # 计算所有向量对的余弦相似度之和
    total_similarity = 0
    count = 0
    
    # 使用itertools.combinations获取所有两两组合
    for i, j in combinations(range(n), 2):
        # 点积计算
        dot_product = np.dot(embeddings[i], embeddings[j])
        
        # 计算余弦相似度
        similarity = dot_product / (norms[i] * norms[j])
        
        total_similarity += similarity
        count += 1
    
    # 计算平均值
    avg_similarity = total_similarity / count
    
    return avg_similarity
def average_clustering_across_graphs(edge_lists):
    """
    输入：多个图的边集列表，每个元素是 [(u, v), ...]
    输出：所有图的平均聚类系数（每个图的聚类系数取平均，再对所有图取平均）
    """
    clustering_values = []
    for edges in edge_lists:
        G = nx.Graph()
        G.add_edges_from(edges.t().cpu().numpy())
        if G.number_of_nodes() < 2:
            clustering = 0.0
        else:
            clustering = nx.average_clustering(G)
        clustering_values.append(clustering)

    if len(clustering_values) == 0:
        return 0.0
    return sum(clustering_values) / len(clustering_values)
def average_triangle_count(edge_lists):
    """
    输入：多个图的边集（每个图是一个边列表）
    输出：这些图中三角形数量的平均值
    """
    triangle_counts = []
    for edges in edge_lists:
        G = nx.Graph()
        G.add_edges_from(edges.t().cpu().numpy())

        # nx.triangles 返回每个节点的三角形数，三角形被3个节点重复统计
        triangle_dict = nx.triangles(G)
        total_triangles = sum(triangle_dict.values()) // 3  # 每个三角形被算3次
        triangle_counts.append(total_triangles)

    avg = sum(triangle_counts) / len(triangle_counts) if triangle_counts else 0
    return avg
def compute_avg_degree_from_edges(edge_lists,syn_x, is_directed=False):
    """
    计算多个图的平均度数（输入为边集）
    
    参数：
        edge_lists: List[List[Tuple[int, int]]]
            多个图的边集，每个图是一个边列表，例如 [(0, 1), (1, 2)]
        is_directed: bool
            是否为有向图
        
    返回：
        List[float]，每个图的平均度数
    """
    avg_degrees = []
    for i in range(len(edge_lists)):
        avg_degrees.append(2*len(edge_lists[i][0])/len(syn_x[i]))
    # for edges in edge_lists:
    #     edges = edges.t()
    #     degree = defaultdict(int)
    #     for u, v in edges:
    #         degree[u] += 1
    #         if not is_directed:
    #             degree[v] += 1
    #     num_nodes = len(degree)
    #     total_degree = sum(degree.values())
    #     avg_degree = total_degree / num_nodes if num_nodes > 0 else 0
    #     avg_degrees.append(avg_degree)
    return avg_degrees
class graph_dataset(Dataset):
    def __init__(self,node_features,labels,edge_indexs):
        self.xs = node_features
        self.ys = labels
        self.edge_indexs = edge_indexs

    def __len__(self):
        return len(self.ys)

    def __getitem__(self,idx):
        x=self.xs[idx]
        y = self.ys[idx]
        edge_index = self.edge_indexs[idx]
        return x,y,edge_index


def get_args():
    parser = argparse.ArgumentParser(description='Train Model')
    parser.add_argument('--data_type', default='DD', type=str,
                        choices=['DD', 'PTC_MR', 'NCI1', 'PROTEINS', 'IMDB-BINARY', 'IMDB-MULTI', 'MUTAG', 'COLLAB'],
                        help='dataset type')
    parser.add_argument('--batch_size', default=200, type=int, help='train batch size')
    parser.add_argument('--num_epochs', default=100, type=int, help='train epochs number')
    parser.add_argument('--seed', default=324, type=int, help='random seed')
    parser.add_argument('--dis_metric',default='cos')
    parser.add_argument('--th',default='0.99')
    parser.add_argument('--device',default='cuda')
    return parser.parse_args()

class syndata:
    def __init__(self,x,edge_index):
        self.x = x     
        self.edge_index = edge_index     
        self.batch =1
if __name__ == '__main__': 
    ipc = 10
    dataset_name = 'PROTEINS'   
    opt = get_args()
    res = []
    #set_determ(opt.seed)
    device = (
        "cuda" if torch.cuda.is_available() else 
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    

    #vis = visdom.Visdom(env=opt.data_type)  # To plot loss and accuracy
    data_set = TUDataset(
        'data/'+dataset_name,
        dataset_name,
        
        #use_node_attr=True,
    )  #pre_transform=Indegree(),
    #data_set = GNNBenchmarkDataset('data/',"CIFAR10")
    print(data_set.num_features)
    test_acc_gcn = 0
    test_acc_gat = 0
    test_acc_dgcnn = 0
    nmi = []
    f1 = []
    max_acc = []
    test_acc_gin = 0
    test_acc_gsage = 0
    for epoch in range(3):
        num_classes = data_set.num_classes
        max_num_nodes = max([x.num_nodes for x in data_set])
        num_features = data_set.num_features
        
        if dataset_name == "COLLAB":
            num_features = 128
        
        #model = GraphSAGE(data_set.num_features,32, data_set.num_classes,2).to(device)
        #model_syn = graphModelforSyn(data_set.num_features, data_set.num_classes).to(device)
        generate_model = GVAE(num_features, 64, 128).to(device)

        params = torch.load('./res/'+dataset_name+'/gvae.pth')
        # 加载参数到模型中
        generate_model.load_state_dict(params)


        # generate_model = generate_model.to(device)
        labels = data_set.y
        _, _tt, idx_train, idx_test = train_test_split(labels, list(range(len(labels))), test_size=0.2,random_state = 42)
        idx_val,idx_test = train_test_split(idx_test,test_size = 0.5 )

        #####################################
        labels_train = labels[idx_train]
        idx_selected = []
        cnt = ipc
        embeds = data_set.x

        for class_id in range(num_classes):
            # print(type(class_id))
            # print(labels_train)
            print(type(idx_train))
            idx_train = torch.tensor(idx_train)
            idx = idx_train[labels_train == class_id]
            feature = embeds[idx]

            # Randomly select initial centroids
            random_indices = np.random.choice(len(feature), size=cnt, replace=False)
            centroids = feature[random_indices]

            for _ in range(300):  # Maximum iterations for convergence
                # Compute distances to centroids
                distances = torch.cdist(feature, centroids)
                cluster_assignments = torch.argmin(distances, dim=1)

                # Update centroids
                for i in range(cnt):
                    assigned_points = feature[cluster_assignments == i]
                    if len(assigned_points) > 0:
                        centroids[i] = torch.mean(assigned_points, dim=0)

            # Randomly select nodes from each cluster
            for i in range(cnt):
                cluster_nodes = idx[(cluster_assignments == i).cpu()]
                if len(cluster_nodes) > 0:
                    selected_node = np.random.choice(
                        cluster_nodes, size=1, replace=False
                    )
                    idx_selected.append(selected_node)
        for i in range(len(idx_selected)):
            idx_selected[i] = idx_selected[i][0]
        print(idx_selected[0])
        print(idx_selected)
        print(len(idx_selected))



    ################################

        loss_fn = nn.NLLLoss()  # Set loss criterion to negative log likelihood loss
        

        dataset_train = data_set[idx_train]
        dataset_val = data_set[idx_val]
        dataset_test = data_set[idx_test]

        train_x = [sample.x for sample in dataset_train]
        train_y = [sample.y for sample in dataset_train]
        train_e = [sample.edge_index for sample in dataset_train]
        avg = compute_avg_degree_from_edges(train_e,train_x)
        print("tt",average_triangle_count(train_e))
        print("平均度数",np.mean(avg))
        print("聚类系数",average_clustering_across_graphs(train_e))
        datalist_train = []
        for i in dataset_train:
            datalist_train.append(Data(x=i.x,edge_index=i.edge_index,y=i.y)) 
        train_loader = DataLoader(dataset=datalist_train,batch_size=opt.batch_size,shuffle=False,collate_fn=collate_fn)
        val_loader = DataLoader(dataset=dataset_val,batch_size=opt.batch_size,shuffle=False,collate_fn=collate_fn)
        test_loader = DataLoader(dataset=dataset_test,batch_size=opt.batch_size,shuffle=False,collate_fn=collate_fn)


        class_idx = [[] for i in range(num_classes)]
    
        for i,data in enumerate(dataset_train):
            class_idx[data.y.item()].append(i)
        dataset_syn = []
        for i in range(num_classes):
            #dataset_syn.extend(dataset_train[list(np.random.permutation(class_idx[i])[:ipc])])
            dataset_syn.extend(data_set[idx_selected])
        #dataset_syn = syn_data_temp
        syn_x_tmp = []
        syn_y = []
        
        syn_x_tmp = [sample.x for sample in dataset_syn]

        syn_y = [sample.y for sample in dataset_syn]
        syn_x =[] 
        print(data_set[0].num_nodes)
        for i in syn_x_tmp:
            #syn_x.append(nn.Parameter(torch.FloatTensor(i.shape[0],i.shape[1]).to(device)))
            syn_x.append(nn.Parameter(i.to(device)))

        syn_e = [sample.edge_index for sample in dataset_syn]
        avg = compute_avg_degree_from_edges(syn_e,syn_x)
        # print(average_triangle_count(syn_e))
        # print("平均度数",np.mean(avg))
        # print("聚类系数",average_clustering_across_graphs(syn_e))

        syn_emb = []
        for i in range(len(syn_e)):
            syn_emb.append(nn.Parameter(generate_model.encode(syn_x[i],syn_e[i].to(device))))
        
        # print(len(syn_e))
        # print(len(syn_emb))

        syn_e_orig = [sample.edge_index for sample in dataset_syn]
        # syn_e = []
        # for xsyn in syn_x:
        #     edgelist=[]
        #     edgelist.append(list(range(len(xsyn))))
        #     edgelist.append(list(range(len(xsyn))))
        #     edgelist = torch.tensor(edgelist).to(device)
        #     syn_e.append(edgelist)

        syn_adj = []
        for i in range(len(syn_x)):
            temp = torch.zeros(len(syn_x[i]),len(syn_x[i]))
            temp[syn_e[i][0]][syn_e[i][1]] = 1
            temp[syn_e[i][1]][syn_e[i][0]] = 1
            syn_adj.append(temp)




        optim_gen = torch.optim.SGD(syn_emb+ list(generate_model.parameters()))


        def get_graphs(c, n): # get random n images from class c
            idx_shuffle = np.random.permutation(class_idx[c])[:n]
            #print(len(idx_shuffle))
            return dataset_train[idx_shuffle]
        n_act = getattr(nn, 'ELU')()


        # optimizer_graph = torch.optim.SGD(syn_x, lr=0.1, momentum=0.5)#syn_x)#[sample.x for sample in dataset_syn])#+[sample.edge_index for sample in dataset_syn]) # optimizer_img for synthetic data
        # optimizer_graph.zero_grad()

        train_loss_list = []
        grad_loss_list = []
        train_acc_list = []
        train_acc_syn = []
        train_acc_org = []
        final_train_loss = []
        final_train_acc = []
        # print(len(syn_e[0][0]))
        # print(len(syn_x[0]))
        max_grad_loss = 10000
        outer_loop = 20
        for train_epoch in range(200):####
            start_time = time.time()
            torch.cuda.reset_peak_memory_stats()
            # peak_mem = torch.cuda.max_memory_allocated() / 1024**2
            # print(f"🎮 一轮训练显存峰值: {peak_mem:.2f} MB")
            model = graphModel(data_set.num_features, data_set.num_classes).to(device) 
            optimizer = Adam(model.parameters()) # Create Adam optimizer for model parameters
            ##生成graph
            for ol in range(outer_loop):
                print(ol)
                # syn_e = []
                # for i in syn_adj:
                #     i[i<0.99] = 0
                #     syn_e.append(torch.nonzero(i).t())
                syn_e = []
    
                syn_x = []
                for emb in syn_emb:
                    tmp_e,tmp_x = generate_model.decode(emb)
                    syn_x.append(nn.Parameter(tmp_x.to(device)))
            
        
                    tmp_e[tmp_e<0.5] = 0
                    syn_e.append(torch.nonzero(tmp_e).t())
            
                loss_dis = torch.tensor(0.0).to(opt.device)
                loss_sum=0
                correct_syn = 0
                cnt = 0
                cnt_orig = 0
                cnt_syn = 0
                correct_org = 0
                pred_0 = []
                for c in range(num_classes):
                    data_real = get_graphs(c,opt.batch_size)
                    data_x_c = [sample.x for sample in data_real]
                    data_y_c = [sample.y for sample in data_real]
                    data_edge_c = [sample.edge_index for sample in data_real]
                    datalist = []
                    for i in range(len(data_x_c)):
                        datalist.append(Data(x=data_x_c[i],edge_index=data_edge_c[i],y=data_y_c[i]))
                    orig_loader = DataLoader(dataset=datalist,batch_size=opt.batch_size,shuffle=False,collate_fn=collate_fn)

                    #train_loss, train_acc = train(real_loader, model, loss_criterion, optimizer, device)
                    data_syn = dataset_syn[c*ipc:c*ipc+ipc]
                    syn_x_c = syn_x[c*ipc:c*ipc+ipc]
                    syn_y_c = syn_y[c*ipc:c*ipc+ipc]
                    syn_e_c = syn_e[c*ipc:c*ipc+ipc]
                    l = 0
                    for j in syn_x_c:
                        l+=len(j)
                        print(l)
                    datalist = []
                    for i in range(len(syn_x_c)):
                        datalist.append(Data(x=syn_x_c[i],edge_index=syn_e_c[i],y=syn_y_c[i]))
                    syn_loader = DataLoader(dataset=datalist,batch_size=opt.batch_size,shuffle=False,collate_fn=collate_fn)
                    loss_dis = torch.tensor(0.0).to(opt.device)
                    for orig,syn in zip(orig_loader,syn_loader):
                        orig = orig.to(device)
                        syn = syn.to(device)
                        pred = model(orig)
                        
                        loss = loss_fn(pred, orig.y)
                        gw_real = torch.autograd.grad(loss, model.parameters())
                        gw_real = list((_.detach().clone() for _ in gw_real))
                        pred_syn = model(syn)
                        # print("pred",len(pred_syn))
                        # print(c,average_cosine_similarity(pred_syn))
                        if c==0:
                            pred_0 = pred_syn
                    
                            #print("cross",average_cross_group_cosine_similarity(pred_0,pred_syn))
                        loss_syn = loss_fn(pred_syn,syn.y)
                        correct = (pred.argmax(dim=1) == orig.y).sum().item()
                        correct_syn += (pred_syn.argmax(dim=1) == syn.y).sum().item()
                        correct_org += correct
                        cnt_syn += len(syn.y)
                        cnt_orig +=len(orig.y)
                        gw_syn = torch.autograd.grad(loss_syn, model.parameters(), create_graph=True)

                        loss_dis += match_loss(gw_syn, gw_real, opt)
                    loss_sum+=loss_dis
                    #loss_dis = torch.tensor(0.0).to(opt.device)
                    #for loop in range(10):
    

                # optimizer_graph.zero_grad()
                # loss_sum.backward() 
                # optimizer_graph.step()
             
                if(loss_sum.item() < max_grad_loss):
                    max_grad_loss = loss_dis.item()
                    torch.save(syn_x,"./res/"+dataset_name+"/syn_x.pth")
                    torch.save(syn_e,"./res/"+dataset_name+"/syn_e.pth")
                    #torch.save(syn_z,"./res/"+dataset_name+"/syn_z.pth")
                optim_gen.zero_grad()
                loss_sum.backward() 
                optim_gen.step() 

                


                        #print(syn_x[0])
                        #print(loss_dis)
                # print("grad_loss",loss_sum.item())
                # print("train_acc_syn",correct_syn/cnt_syn) 
                # print("trian_acc_org",correct_org/cnt_orig)
                grad_loss_list.append(loss_sum.item()) 
                train_acc_syn.append(correct_syn/cnt_syn) 
                train_acc_org.append(correct_org/cnt_orig) 
                # optimizer_graph.zero_grad()
                # loss_dis.backward()
                # optimizer_graph.step()
                # print(syn_x[0])

                # for ii in range(10):
                loss_tr = 0
                correct = 0
                cnt = 0
                for sample in train_loader:
                    
                    sample = sample.to(device)
                    pred = model(sample)
                    correct += (pred.argmax(dim=1) == sample.y).sum().item()
                    cnt += len(sample.y)
                    loss_tr += loss_fn(pred, sample.y)
                
                optimizer.zero_grad()
                
                loss_tr.backward()
                optimizer.step()
                datalist=[]
                for i in range(len(syn_x)):
                    datalist.append(Data(x=syn_x[i],edge_index=syn_e[i],y=syn_y[i])) 
                syn_loader_all = DataLoader(dataset=datalist,batch_size=opt.batch_size,shuffle=False,collate_fn=collate_fn)
                #for ii in range(10):

                loss_tr = 0
                correct = 0
                cnt = 0
                for sample in syn_loader_all:
                    sample = sample.to(device)
                    pred = model(sample)
                    correct += (pred.argmax(dim=1) == sample.y).sum().item()
                    cnt += len(sample.y)
                    loss_tr += loss_fn(pred, sample.y)
                optimizer.zero_grad()
                loss_tr.backward()
                optimizer.step()
            end_time = time.time()
            epoch_duration = end_time - start_time
            print(f"Epoch time: {epoch_duration:.4f} seconds")
            peak_mem = torch.cuda.max_memory_allocated() / 1024**2
            print(f"🎮 一轮训练显存峰值: {peak_mem:.2f} MB")    

            print(train_epoch)
            # print("train_loss",loss_tr.item())
            # print("train_acc",correct/cnt)
            train_loss_list.append(loss_tr.item())
            train_acc_list.append(correct/cnt)

    #######################3
            if(train_epoch%10 ==0):
                deg = compute_avg_degree_from_edges(syn_e,syn_x)
                print(deg)
                print("平均度数",np.mean(deg))
                print(average_triangle_count(syn_e))
                print("聚类系数",average_clustering_across_graphs(syn_e))
               # print(cross_graph_overlap(syn_e))
                syn_x = torch.load("./res/"+dataset_name+"/syn_x.pth")
                syn_e = torch.load("./res/"+dataset_name+"/syn_e.pth")
                print("len",len(syn_e))
                print(average_triangle_count(syn_e[0:5]))
                print(average_triangle_count(syn_e[5:10]))

                #syn_z = torch.load("./res/"+dataset_name+"/syn_z.pth")
                #idx_shuffle = np.random.permutation(range(len(syn_x)))[:len(syn_x)]

                # np.random.seed(1)
                # np.random.shuffle(syn_x)
                # np.random.shuffle(syn_y)
                # np.random.shuffle(syn_e)
                for j in range(10):
                    model_test = graphModel(data_set.num_features, data_set.num_classes).to(device)
                    loss_fn = nn.CrossEntropyLoss() 
                    optimizer = Adam(model_test.parameters())
                    max_acc_val = 0
                    for i in range(200):
                        loss_test = 0
                        correct = 0
                        cnt = 0
                        correct_val = 0
                        cnt_val = 0 
                        for sample in syn_loader_all:
                            sample = sample.to(device)
                            pred = model_test(sample)
                            correct += (pred.argmax(dim=1) == sample.y).sum().item()
                            cnt += len(sample.y)
                            loss_test += loss_fn(pred, sample.y)
            
                        for sample in val_loader:
                            sample = sample.to(device)
                            pred = model_test(sample)
                            correct_val += (pred.argmax(dim=1) == sample.y).sum().item()
                            cnt_val += len(sample.y)
                    
                        if(correct_val/cnt_val > max_acc_val):
                            max_acc_val = correct_val/cnt_val
                            torch.save(model_test.state_dict(),'./res/'+dataset_name+'/model_best_dgcnn.pth')
                        optimizer.zero_grad()
                        loss_test.backward()
                        optimizer.step()
                        print(i,loss_test.item(),correct/cnt)
                        final_train_acc.append(correct/cnt)
                        final_train_loss.append(loss_test.item())


                    correct = 0
                    correct2 =0
                    cnt = 0

                    params = torch.load('./res/'+dataset_name+'/model_best_dgcnn.pth')
                
                    model_test.load_state_dict(params)
                    for sample in test_loader:
                        sample = sample.to(device)
                        pred = model_test(sample)
                        correct += (pred.argmax(dim=1) == sample.y).sum().item()
                        cnt += len(sample.y)
                        # m_nmi = jaccard_score(pred.argmax(dim=1).cpu().numpy(),sample.y.cpu().numpy())
                        # m_f1 = f1_score(pred.argmax(dim=1).cpu().numpy(),sample.y.cpu().numpy(), average='weighted')
                    print("final dgcnn")
        
                    res.append(correct/cnt)
                    print(get_res(res))
                    
                    test_acc_dgcnn += correct/cnt
                    max_acc.append(correct/cnt)
                    # nmi.append(m_nmi)
                    # f1.append(m_f1)
    print(max_acc)
    print(nmi)
    print(f1)
################################################

        

#         max_acc_val = 0
#         model_test = GAT(data_set.num_features,32, data_set.num_classes,2).to(device)
#         #nfeat, nhid, nclass, dropout, alpha, nheads    
#         loss_fn = nn.CrossEntropyLoss() 
#         optimizer_test = Adam(model_test.parameters())
#         for i in range(100):
#             loss_test = 0
#             correct = 0
#             cnt = 0
#             correct_val = 0
#             cnt_val = 0 
#             for sample in syn_loader_all:
#                 sample = sample.to(device)
#                 pred = model_test(sample)
#                 correct += (pred.argmax(dim=1) == sample.y).sum().item()
#                 cnt += len(sample.y)
#                 loss_test += loss_fn(pred, sample.y)
 
   
#                 #放里面放外面
#             for sample in val_loader:
#                 sample = sample.to(device)
#                 pred = model_test(sample)
#                 correct_val += (pred.argmax(dim=1) == sample.y).sum().item()
#                 cnt_val += len(sample.y)
#             if(correct_val/cnt_val > max_acc_val):
#                 max_acc_val = correct_val/cnt_val
#                 torch.save(model_test.state_dict(), './res/'+dataset_name+'/model_best_gat.pth')
#             optimizer_test.zero_grad()
#             loss_test.backward()
#             optimizer_test.step()
#             print(i,loss_test.item(),correct/cnt)
#         correct = 0
#         correct2 =0
#         cnt = 0
#         params = torch.load('./res/'+dataset_name+'/model_best_gat.pth')
#         # 加载参数到模型中
#         model_test.load_state_dict(params)
#         for sample in test_loader:
#             sample = sample.to(device)
#             pred = model_test(sample)
#             correct += (pred.argmax(dim=1) == sample.y).sum().item()
#             cnt += len(sample.y)

#         print("final gat")
#         print(correct/cnt)

#         test_acc_gat +=correct/cnt
# ##############################
#         max_acc_val = 0
#         model_test = GCN(data_set.num_features,32, data_set.num_classes,2).to(device)
#         #nfeat, nhid, nclass, dropout, alpha, nheads    
#         loss_fn = nn.CrossEntropyLoss() 
#         optimizer_test = Adam(model_test.parameters())
#         for i in range(100):
#             loss_test = 0
#             correct = 0
#             cnt = 0
#             correct_val = 0
#             cnt_val = 0 
#             for sample in syn_loader_all:
#                 sample = sample.to(device)
#                 pred = model_test(sample)
#                 correct += (pred.argmax(dim=1) == sample.y).sum().item()
#                 cnt += len(sample.y)
#                 loss_test += loss_fn(pred, sample.y)
 
   
#                 #放里面放外面
#             for sample in val_loader:
#                 sample = sample.to(device)
#                 pred = model_test(sample)
#                 correct_val += (pred.argmax(dim=1) == sample.y).sum().item()
#                 cnt_val += len(sample.y)
#             if(correct_val/cnt_val > max_acc_val):
#                 max_acc_val = correct_val/cnt_val
#                 torch.save(model_test.state_dict(), './res/'+dataset_name+'/model_best_gcn.pth')
#             optimizer_test.zero_grad()
#             loss_test.backward()
#             optimizer_test.step()
#             print(i,loss_test.item(),correct/cnt)
#         correct = 0
#         correct2 =0
#         cnt = 0
#         params = torch.load('./res/'+dataset_name+'/model_best_gcn.pth')
#         # 加载参数到模型中
#         model_test.load_state_dict(params)
#         for sample in test_loader:
#             sample = sample.to(device)
#             pred = model_test(sample)
#             correct += (pred.argmax(dim=1) == sample.y).sum().item()
#             cnt += len(sample.y)

#         print("final gcn")
#         print(correct/cnt)

#         test_acc_gcn +=correct/cnt
# ################################################
#         max_acc_val = 0
#         model_test = GIN(data_set.num_features,32, data_set.num_classes,2).to(device)
#         #nfeat, nhid, nclass, dropout, alpha, nheads    
#         loss_fn = nn.CrossEntropyLoss() 
#         optimizer_test = Adam(model_test.parameters())
#         for i in range(100):
#             loss_test = 0
#             correct = 0
#             cnt = 0
#             correct_val = 0
#             cnt_val = 0 
#             for sample in syn_loader_all:
#                 sample = sample.to(device)
#                 pred = model_test(sample)
#                 correct += (pred.argmax(dim=1) == sample.y).sum().item()
#                 cnt += len(sample.y)
#                 loss_test += loss_fn(pred, sample.y)
 
   
#                 #放里面放外面
#             for sample in val_loader:
#                 sample = sample.to(device)
#                 pred = model_test(sample)
#                 correct_val += (pred.argmax(dim=1) == sample.y).sum().item()
#                 cnt_val += len(sample.y)
#             if(correct_val/cnt_val > max_acc_val):
#                 max_acc_val = correct_val/cnt_val
#                 torch.save(model_test.state_dict(), './res/'+dataset_name+'/model_best_gin.pth')
#             optimizer_test.zero_grad()
#             loss_test.backward()
#             optimizer_test.step()
#             print(i,loss_test.item(),correct/cnt)
#         correct = 0
#         correct2 = 0
#         cnt = 0
#         params = torch.load('./res/'+dataset_name+'/model_best_gin.pth')
#         # 加载参数到模型中
#         model_test.load_state_dict(params)
#         for sample in test_loader:
#             sample = sample.to(device)
#             pred = model_test(sample)
#             correct += (pred.argmax(dim=1) == sample.y).sum().item()
#             cnt += len(sample.y)

#         print("final gin")
#         print(correct/cnt)

#         test_acc_gin +=correct/cnt
# ############################################
#         max_acc_val = 0
#         model_test = GraphSAGE(data_set.num_features,32, data_set.num_classes,2).to(device)
#         #nfeat, nhid, nclass, dropout, alpha, nheads    
#         loss_fn = nn.CrossEntropyLoss() 
#         optimizer_test = Adam(model_test.parameters())
#         for i in range(100):
#             loss_test = 0
#             correct = 0
#             cnt = 0
#             correct_val = 0
#             cnt_val = 0 
#             for sample in syn_loader_all:
#                 sample = sample.to(device)
#                 pred = model_test(sample)
#                 correct += (pred.argmax(dim=1) == sample.y).sum().item()
#                 cnt += len(sample.y)
#                 loss_test += loss_fn(pred, sample.y)
 
   
#                 #放里面放外面
#             for sample in val_loader:
#                 sample = sample.to(device)
#                 pred = model_test(sample)
#                 correct_val += (pred.argmax(dim=1) == sample.y).sum().item()
#                 cnt_val += len(sample.y)
#             if(correct_val/cnt_val > max_acc_val):
#                 max_acc_val = correct_val/cnt_val
#                 torch.save(model_test.state_dict(), './res/'+dataset_name+'/model_best_gsage.pth')
#             optimizer_test.zero_grad()
#             loss_test.backward()
#             optimizer_test.step()
#             print(i,loss_test.item(),correct/cnt)
#         correct = 0
#         correct2 = 0
#         cnt = 0
#         params = torch.load('./res/'+dataset_name+'/model_best_gsage.pth')
#         # 加载参数到模型中
#         model_test.load_state_dict(params)
#         for sample in test_loader:
#             sample = sample.to(device)
#             pred = model_test(sample)
#             correct += (pred.argmax(dim=1) == sample.y).sum().item()
#             cnt += len(sample.y)

#         print("final gsage")
#         print(correct/cnt)

#         test_acc_gsage +=correct/cnt


    print("final dgcnn acc",test_acc_dgcnn/3)
    print("final gat acc",test_acc_gat/3)
    print("final gcn acc",test_acc_gcn/3)
    print("final gin acc",test_acc_gin/3)
    print("final gsage acc",test_acc_gsage/3)
        # train_loss_list = []
        # grad_loss_list = []
        # train_acc_list = []
        # train_acc_syn = []
        # final_train_loss = []
        # final_train_acc = []
        # print("train_acc_origin")
        # print(train_acc_org)
        # print("train_loss_list")
        # print(train_loss_list)
        # print("grad_loss_list")
        # print(grad_loss_list)
        # print("train_acc_list")
        # print(train_acc_list)
        # print("train_acc_syn")
        # print(train_acc_syn)
        # print("final_train_loss")
        # print(final_train_loss)
        # print("final_train_acc")
        # print(final_train_acc)








            # gw_real = torch.autograd.grad(loss_real, net_parameters)
            # gw_real = list((_.detach().clone() for _ in gw_real))













