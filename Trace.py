import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Function
from torch.nn import functional as F
import math

def spike_fn(U, threshold):
    return (U >= threshold).float()


#input X spikes is determined by the probability,
#higher the pixel tensity, higher the probability there is a spike.
def poisson_encode(x, T):
    x = x.unsqueeze(0).repeat(T, 1, 1)  # (T, B, features)
    spikes = (torch.rand_like(x) < x).float()
    return spikes


#Now, I implement it with eligibility, because with the last learning window approach, the networks was enforced to
#produce spikes at the ends, however there was no temporal information.
#So, we either use Backpropagation through time, or eligibility trace.
#backpropagation through time is not biologically plausible, so we use eligibility trace.
#Also, a good thing is that eligibility trace and backpropagation through time is
#roughly equivalent to each other mathematically(roughly, not exactly, works in practice).
#eligibility trace basically -> you put a trace to the neurons at each time stop, basicaly it makes the neuron know the
#gradients of the previous time step. So it is why it is equivalent to backpropagation through time.
#each time step we multiply that trace from the previous time step with a number, so it may cause vanishing gradients if
#that factor(gamma) is < 1 , or exploding gradients if gamma > 1, 
#it is used for holding the temporal context.
#also one last thing to note it makes it online, so no need to hold the history, less memory need and biologicaly more plausible
class LIFLayerEligibility(nn.Module):
    def __init__(self, in_features, out_features, tau_trace = 20.0,
                 tau_mem=20.0, tau_syn=5.0, dt=1.0, threshold=1.0, super_spike_B = 0.03):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.threshold = threshold
        self.super_spike_B = super_spike_B
        # use Xavier initialization, because otherwise some neurons were not firing at all, it was unstable.
        std = 1 / (in_features**0.5)
        self.W = nn.Parameter(torch.randn(out_features, in_features) * std)
        

        self.dt = dt #just a theoretical constant

        #Use register_buffer, because it must be linked to the nn.Module
        self.register_buffer('alpha', torch.exp(torch.tensor(-dt / tau_syn))) #Voltage decay
        self.register_buffer('beta', torch.exp(torch.tensor(-dt / tau_mem))) #Current decay
        self.register_buffer('gamma', torch.exp(torch.tensor(-dt / tau_trace)))  #Trace decay


    #input spikes -> (T, B, in_dim)
    #returns -> (T, B, out_dim) as spikes, and (T, B, out_dim) as voltage values
    def forward(self, input_spikes):

        T, B, _ = input_spikes.shape
        device = self.W.device

        #current, and voltage must start from 0 in the beginning
        I = torch.zeros(B, self.out_features, device=device)
        U = torch.zeros(B, self.out_features, device=device)

        #keep all spike and voltage values across the time steps.
        out_spikes = []
        U_hist = []
        trace = torch.zeros(B, self.out_features, self.in_features, device=device)
        for t in range(T):
            x_t = input_spikes[t]  # (B, in_features), X value of the current time step.

            #new current
            I = self.alpha * I + x_t @ self.W.t()  # (B, out_features)

            #new membrane voltage
            U = self.beta * U + I

            #if U exceeds threshold get one as a spike.
            S = spike_fn(U, self.threshold)  # (B, out_features)
            U_hist.append(U.clone())
            U = U - S * self.threshold #reset U
            # Surrogate gradient
            du = (U - self.threshold).abs()
            sigma_prime = 1 / ((1 + self.super_spike_B * du)**2)  # (B, out_features)

            # Eligibility trace update
            # calculate trace here
            trace = self.gamma * trace + sigma_prime.unsqueeze(2) * x_t.unsqueeze(1)
            
            U_hist.append(U.clone())
            U = U - S * self.threshold
            out_spikes.append(S)

            

        out_spikes = torch.stack(out_spikes, dim=0)
        U_hist = torch.stack(U_hist, dim=0)
        return out_spikes, U_hist, trace

class RandomBPSNNEligibility(nn.Module):
    def __init__(self, T=20):
        super().__init__()
        self.T = T
        self.lif_layers = torch.nn.ModuleList()
        self.G_hiddens = torch.nn.ParameterList()


    def append_LIF(self, LIF, out_dim = 10):
        self.lif_layers.append(LIF)
        if (len(self.lif_layers) > 1):
            hidden_dim = self.lif_layers[-2].out_features #set the projection matrix
            #random projection matrix -> (out_dim(10), hidden_dim)
            G_hidden = nn.Parameter(
                torch.randn(out_dim, hidden_dim) * 0.1, requires_grad = False
            )
            self.G_hiddens.append(G_hidden)

    #here if X is a static no temporal data -> X values are encoded using rate-encoding or poisson encoding
    #generally rate-encoding is used because it is more stable however I implemented poisson encoding above as well.
    #if it has a stime steps, no encoding is used.

    #Static X -> (B, in_dim)
    #Time_step_X -> (T, B, in_dim)
    def forward(self, x):
        device = self.lif_layers[0].W.device
        static = len(x.shape) == 2
        x = x.to(device)
        if (static):
            B, in_dim = x.shape
            print("????")
            x = x.unsqueeze(0).repeat(self.T, 1, 1)  # (T, B, in_dim), rate encoding

        else:
            _, B, in_dim = x.shape
        o_spk = x
        spikes, voltages, traces = [], [], []
        for lif in self.lif_layers:
            o_spk, o_U, trace = lif(o_spk)
            spikes.append(o_spk)
            voltages.append(o_U)
            traces.append(trace)

        #to calculate accuracy, we generally sum spike numbers of all the time steps, and predict the 
        #class which has maximum number of summed spikies.
        out_rate = o_spk.mean(dim=0)  #(B, O)

        return {
            "Us": voltages,
            "spikes": spikes,
            "o_spk": spikes[-1],
            "o_U": voltages[-1],
            "out_rate": out_rate,
            "traces": traces
        }
the_device = torch.device("gpu")
def random_bp_step_eligibility(model, x, target, optimizer):
    """
    model -> all_layers(RandomBPSNN)
    x -> (B, in_dim) or (T, B, in_dim) if it is (B, in_dim) timesteps are produced using rate-encoding
    target -> (B, out_dim)
    loss_fn -> Cross entropy derivative is used here -> |Y - y_predicted|
    """    

    optimizer.zero_grad()
    x = x.to(the_device)
    out = model(x)
    spikes = out["spikes"] # (layer_num, T, B, O)
    traces = out["traces"] # (layer_num, ) -> trace values of the last time step for each layers, 
    out_spikes = out["o_spk"] # (T, B, 10)
    T, B, num_classes = out_spikes.shape
    
    #apply one_hot encoding if it hasnt applied yet.
    if (len(target.shape) == 1):
        target = F.one_hot(target, num_classes=num_classes).float()  # (B, 10)

    #calculate errors
    out_rate = spikes[-1].mean(dim=0)  # (B, num_classes)
    error = out_rate - target           # (B, num_classes)
    loss = F.cross_entropy(out_rate, target)

    #calculate gradients from the traces
    for i, lif in enumerate(model.lif_layers):
        trace = traces[i]  #(B, out, in)
        
        #if it is output layer just use normal error value, if it is a hidden layer use projection of the error derivative
        if i == len(model.lif_layers) - 1:
            #Output layer -> dW = error * trace
            dW = torch.einsum('bo,boi->oi', error, trace) / B
        else:
            #Hidden layer: project error with random matrix
            G = model.G_hiddens[i]
            proj_error = error @ G  # (B, hidden)
            dW = torch.einsum('bh,bhi->hi', proj_error, trace) / B
        
        lif.W.grad = dW
    
    optimizer.step()
    return loss.item()


#train the model, with eligibility applied
def train_eligibility(train_loader, test_loader, model, optimizer, num_epochs, verbose = True, apply_poisson = False, T = 20, dont_touch = True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    spike_means_outer = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)   # (B,1,28,28)
            target = target.to(device)
            if not dont_touch:
                # Flatten: (B, 784)
                data = data.view(data.size(0), -1)

                if (not apply_poisson):
                # Flatten: (B, 784)
                    data = data.unsqueeze(0).repeat(T, 1, 1)  # (T, B, in_dim), rate encoding
                else:
                    data = poisson_encode(data, T = T)
                    

            total_loss += random_bp_step_eligibility(model, data, target, optimizer)

            if (batch_idx + 1) % 8 == 0:
                if (verbose):
                    print(f"Epoch {epoch} | Batch {batch_idx+1}/{len(train_loader)} | "
                        f"Loss: {total_loss / (batch_idx+1):.4f}")
                data, target = next(iter(train_loader))
                if (not dont_touch):
                    data = data.to(device).view(data.size(0), -1)

                out = model(data)
                spike_means = []
                for i in out["spikes"]:
                    spike_means.append(i.mean().item())
                spike_means_outer.append(torch.tensor(spike_means))
                if (verbose):
                    for i in range(len(spike_means)):
                        print(f"layer {i+1} spike means:", spike_means[i])


        #calculate accuracy
        acc = evaluate(model, test_loader, device)
        if (verbose):
            print(f"Epoch {epoch} finished. Test accuracy: {acc:.2f}%")
    spike_means_outer = torch.stack(spike_means_outer, dim = 0)
    return model, acc, spike_means_outer
def evaluate(model, loader, device, dont_touch = True):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            target = target.to(device)
            if (not dont_touch):
                data = data.view(data.size(0), -1) #it is not already flattened, so flatten it.

            out = model(data)
            out_rate = out["out_rate"]  # (B, out_dim)
            pred = out_rate.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    return 100.0 * correct / total


device = torch.device("cuda")
#returns a model and optimizer,
#you can determine the hidden layer number with hidden_dims
#hidden dims -> [X, Y, Z, T, U] it means that input layer is X; Y, Z, T are neuron numbers of the hidden layers, and U is out_dim
#you can set different thresholds for each of the hidden_layer, default is 1 threshold
def get_model_eligibility(hidden_dims, thresholds = None, super_spike_beta = 25, lr = 1e-3, out_dims = 10):
    if (thresholds is None):
        thresholds = [1 for _ in range(len(hidden_dims) - 1)]

    model = RandomBPSNNEligibility()
    weights = []
    for i in range(1, len(hidden_dims)):
        model.append_LIF(LIFLayerEligibility(hidden_dims[i-1], hidden_dims[i], threshold=thresholds[i-1], super_spike_B=super_spike_beta), out_dim=out_dims)
        weights.append(model.lif_layers[-1].W)
    model = model.to(device)
    optimizer = torch.optim.Adam(
        weights, lr=lr
    )
    return model, optimizer