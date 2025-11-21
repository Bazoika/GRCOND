import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from ogb.graphproppred.mol_encoder import AtomEncoder
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_add_pool, global_mean_pool
from tqdm.notebook import tqdm

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        
  
        self.fc1 = nn.Linear(input_dim, 64)
        
   
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)
        

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.Sigmoid()  
        )
    
    def encode(self, x):

        h = F.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):

        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss(recon_x, x, mu, logvar):
 
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def train_vae(model, dataloader, epochs=50, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            inputs = batch.x
            
            
            recon_batch, mu, logvar = model(inputs)
            
          
            loss = vae_loss(recon_batch, inputs, mu, logvar)
          
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader.dataset):.4f}')

class GraphTransformerEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, latent_dim):
        super(GraphTransformerEncoder, self).__init__()
        self.conv1 = TransformerConv(in_channels, hidden_dim)
        self.conv2 = TransformerConv(hidden_dim, latent_dim * 2)  # 2*latent_dim for mean & log_var

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        mu, log_var = torch.chunk(x, 2, dim=1)  
        return mu, log_var

class GraphDecoder(nn.Module):
    def __init__(self, latent_dim):
        super(GraphDecoder, self).__init__()
        self.linear = nn.Linear(latent_dim, latent_dim)
        self.linear2 = nn.Linear(latent_dim,latent_dim)
        self.bn = nn.BatchNorm1d(latent_dim) 

    def forward(self, z):
        z = self.linear(z) 
        #z = self.bn(z)  
        # z = F.relu(z)  
        # z = self.linear2(z)
        adj_recon = torch.sigmoid(torch.matmul(z, z.T))  
        return adj_recon

        
class GVAE(nn.Module):
    def __init__(self, in_channels, hidden_dim, latent_dim):
        super(GVAE, self).__init__()
        self.encoder = GraphTransformerEncoder(in_channels, hidden_dim, latent_dim)
        self.decoder = GraphDecoder(latent_dim)
        self.decoder_x = nn.Linear(latent_dim,in_channels)
        self.decoder2 = nn.Linear(64,in_channels)
        self.decoder1 = nn.Linear(latent_dim,64)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std  # 采样

    def forward(self, x, edge_index):
        mu, log_var = self.encoder(x, edge_index)
        z = self.reparameterize(mu, log_var)
        adj_recon = self.decoder(z)
        # x_recon = F.relu(self.decoder1(z))
        # x_recon = torch.sigmoid(self.decoder2(x_recon))
        x_recon = torch.sigmoid(self.decoder_x(z))
        return adj_recon,x_recon, mu, log_var
    
    def encode(self,x,edge_index):
        mu, log_var = self.encoder(x, edge_index)
        z = self.reparameterize(mu, log_var)
        return z

    def decode(self,z):
        adj_recon = self.decoder(z)
        x_recon = torch.sigmoid(self.decoder_x(z))
        return adj_recon,x_recon

    def loss_function(self, adj_recon, adj, mu, log_var,x,x_recon):
        recon_loss = F.mse_loss(adj_recon, adj)
        recon_loss_x = F.mse_loss(x_recon,x)
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        return recon_loss + kl_loss +recon_loss_x


if __name__ == '__main__': 
    dataset_name = 'ogbg_molbbbp'
    #dataset = TUDataset(root='../graphcond/data/'+dataset_name, name=dataset_name)
    dataset = PygGraphPropPredDataset(root='../graphcond/data/'+dataset_name,name='ogbg-molbbbp')
    print(dataset.num_features)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_channels = dataset.num_features
    print(in_channels)
    hidden_dim = 64
    latent_dim = 128
    model = GVAE(in_channels, hidden_dim, latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # vae = VAE(input_dim=in_channels, latent_dim=32)

    # train_vae(vae,dataloader,50)



    for epoch in range(30):
        model.train()
        total_loss = 0
        tloss = 0
        optimizer.zero_grad()
        for data in dataloader:
            #print(type(data.x))
            data = data.to(device)
            #data.x = torch.FloatTensor(data.x)
            adj = torch.matmul(data.x.float(), data.x.T.float())  
            
            
            adj_recon,x_recon, mu, log_var = model(data.x.float(), data.edge_index)
            # print(x_recon.shape)
            # print(adj_recon.shape)
            #loss = model.loss_function(adj_recon, adj, mu, log_var,data.x.float(),x_recon)
            cos_sim = torch.nn.functional.cosine_similarity(adj,adj_recon, dim=-1)

            loss = 1 - cos_sim.mean()
            cos_sim = torch.nn.functional.cosine_similarity(data.x,x_recon)
            loss +=1-cos_sim.mean()
            tloss += loss
        tloss.backward()
        optimizer.step()
        total_loss += tloss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader):.4f}")
    torch.save(model.state_dict(),'../graphcond/res/'+dataset_name+'/gvae.pth')