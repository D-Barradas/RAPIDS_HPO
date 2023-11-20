import skorch.utils
from skorch import NeuralNetRegressor
import torch.nn as nn
import torch
import skorch


def _initialize(method, layer, gain=1):
    weight = layer.weight.data
    # _before = weight.data.clone()
    kwargs = {'gain': gain} if 'xavier' in str(method) else {}
    method(weight.data, **kwargs)
    # assert torch.all(weight.data != _before)


class Autoencoder(nn.Module):
    def __init__(self, activation='ReLU', init='xavier_uniform_',
                 **kwargs):
        super().__init__()

        self.activation = activation
        self.init = init
        self._iters = 0

        init_method = getattr(torch.nn.init, init)
        act_layer = getattr(nn, activation)
        act_kwargs = {'inplace': True} if self.activation != 'PReLU' else {}

        gain = 1
        if self.activation in ['LeakyReLU', 'ReLU']:
            name = 'leaky_relu' if self.activation == 'LeakyReLU' else 'relu'
            gain = torch.nn.init.calculate_gain(name)

        inter_dim = 28 * 28
        latent_dim = inter_dim // 4
        layers = [
            nn.Linear(28 * 28, inter_dim),
            act_layer(**act_kwargs),
            nn.Linear(inter_dim, latent_dim),
            act_layer(**act_kwargs)
        ]
        for layer in layers:
            if hasattr(layer, 'weight') and layer.weight.data.dim() > 1:
                _initialize(init_method, layer, gain=gain)
        self.encoder = nn.Sequential(*layers)
        layers = [
            nn.Linear(latent_dim, inter_dim),
            act_layer(**act_kwargs),
            nn.Linear(inter_dim, 28 * 28),
            nn.Sigmoid()
        ]
        for layer in layers:
            if hasattr(layer, 'weight') and layer.weight.data.dim() > 1:
                _initialize(init_method, layer, gain=gain)
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        self._iters += 1
        shape = x.size()
        # x = x.squeeze(1)  # Add this line
        x = x.view(x.shape[0], -1)
        x = self.encoder(x)
        x = self.decoder(x)
        return x.view(shape)
    
class NegLossScore(NeuralNetRegressor):
    steps = 0
    def partial_fit(self, *args, **kwargs):
        super().partial_fit(*args, **kwargs)
        self.steps += 1
        
    def score(self, X, y):
        X = skorch.utils.to_tensor(X, device=self.device)
        y = skorch.utils.to_tensor(y, device=self.device)

        ## Add the input.squeeze() here
        X = X.squeeze(1)
        y = y.squeeze(1)

        
        # print("This is the shape of X tensor:",X.shape)
        # print("This is the shape of y tensor:",y.shape)

        
        self.initialize_criterion()
        y_hat = self.predict(X)

        # y_hat = y_hat.squeeze(1)  ## Add the squeeze() here before the view()
        # y_hat = y_hat.view(12600, 784)  ## Add the input.squeeze() here

        # print("This is the shape of y_hat tensor:",y_hat.shape)
        # y_hat = y_hat.view(12600, 784) ## Add the input.squeeze() here



        y_hat = skorch.utils.to_tensor(y_hat, device=self.device)
        # print("This is the shape of y_hat tensor:",y_hat.shape)

        # the tensor y_hat is a 3D tensor with the shape (12600, 2, 784). 
        # This type of tensor is often used to represent images, 
        # where the first dimension represents the number of images, 
        # the second dimension represents the number of channels (e.g., red, green, blue), 
        # and the third dimension represents the number of pixels in each channel
        # in this case we have 2 channels (black and white)
        # we will use only one channel as it will contain already the information 
        # nesesary to run the code 
        y_hat = y_hat.squeeze(1)  ## Add the squeeze() here before the view()
        y_hat = y_hat.view(-1, 784)
        y_hat = y_hat[0:12600, :]


        # print("This is the shape of y_hat tensor after skorch :",y_hat.shape)
        # print("This is the type of y_hat tensor after skorch :",type(y_hat))


        loss = super().get_loss(y_hat, y, X=X, training=False).item()
        print(f'steps = {self.steps}, loss = {loss}')
        return -1 * loss
    
    def initialize(self, *args, **kwargs):
        super().initialize(*args, **kwargs)
        self.callbacks_ = []
        
