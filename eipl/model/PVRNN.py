from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from eipl.layer import PVRNNCell


class PVRNNForTrain(nn.Module):
    def __init__(self,
                 z_dim,
                 d_dim,
                 w,
                 joint_dim=8):
        super(PVRNNForTrain, self).__init__()
        
        self.rnn = PVRNNCell(z_dim, d_dim, w)
        
        decoder_in = d_dim + z_dim
        self.decoder_joint = nn.Sequential(nn.Linear(decoder_in, joint_dim), nn.ReLU(True))
        
        self.decoder_image = nn.Sequential(
            nn.Linear(decoder_in, 8 * 4 * 4),
            nn.LayerNorm([8 * 4 * 4]),
            nn.ReLU(True),
            nn.Unflatten(1, (8, 4, 4)),
            nn.ConvTranspose2d(8, 12, 3, 2, padding=1, output_padding=1),
            nn.LayerNorm([12, 8, 8]),
            nn.ReLU(True),
            nn.ConvTranspose2d(12, 16, 3, 2, 1, 1),
            nn.LayerNorm([16, 16, 16]),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 32, 3, 2, 1, 1),
            nn.LayerNorm([32, 32, 32]),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 64, 3, 2, 1, 1),
            nn.LayerNorm([64, 64, 64]),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 3, 2, 1, 1),
            nn.ReLU(True),
        )
    
    def forward(self, A, state=None):
        z, d, kld, wkld = self.rnn(A, state)
        zd = torch.concat([z, d], dim=1)
        y_joint = self.decoder_joint(zd)
        y_image = self.decoder_image(zd)
        
        return y_image, y_joint, kld, wkld, d


class PVRNNForTest(nn.Module):
    def __init__(self,
                 z_dim,
                 d_dim,
                 w,
                 window_size,
                 A_lr,
                 update_steps,
                 joint_dim=8):
        super(PVRNNForTest, self).__init__()
        
        self.window_size = window_size
        
        self.rnn = PVRNNCell(z_dim, d_dim, w)
        
        decoder_in = d_dim + z_dim
        self.decoder_joint = nn.Sequential(nn.Linear(decoder_in, joint_dim), nn.ReLU(True))
        
        self.decoder_image = nn.Sequential(
            nn.Linear(decoder_in, 8 * 4 * 4),
            nn.LayerNorm([8 * 4 * 4]),
            nn.ReLU(True),
            nn.Unflatten(1, (8, 4, 4)),
            nn.ConvTranspose2d(8, 12, 3, 2, padding=1, output_padding=1),
            nn.LayerNorm([12, 8, 8]),
            nn.ReLU(True),
            nn.ConvTranspose2d(12, 16, 3, 2, 1, 1),
            nn.LayerNorm([16, 16, 16]),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 32, 3, 2, 1, 1),
            nn.LayerNorm([32, 32, 32]),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 64, 3, 2, 1, 1),
            nn.LayerNorm([64, 64, 64]),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 3, 2, 1, 1),
            nn.ReLU(True),
        )
        
        self.mse_loss = nn.MSELoss()
        self.A = torch.zeros(window_size, 1, z_dim*2)
        self.A_opt = optim.SGD([self.A,], lr=A_lr)
        self.update_steps = update_steps
        
        self.window = deque(maxlen=window_size)
    
    def forward(self, xi, xv, state=None):
        self.window.append([xi, xv])
        
        for _ in range(self.update_steps):
            d = state
            vfe = 0
            self.A_opt.zero_grad(True)
            for wt in range(len(self.window)):
                wi, wv = self.window[wt]
                A = self.A[wt]
                z, d, _, wkld = self.rnn(A, d)
                zd = torch.concat([z, d], dim=1)
                yi = self.decoder_image(zd)
                yv = self.decoder_joint(zd)
                mse = self.mse_loss(yi, wi) + self.mse_loss(yv, wv)
                vfe = vfe + mse + wkld
            vfe.backward()
            self.A_opt.step()

        with torch.no_grad():
            d = state
            for wt in range(len(self.window)):
                wi, wv = self.window[wt]
                A = self.A[wt]
                z, d, _, wkld = self.rnn(A, d)
                if len(self.window == self.window_size) and wt == 0:
                    state = d.detach()
            if len(self.window == self.window_size):
                self.A[:-1] = self.A[1:].clone()
                self.A[-1] = 0
            next_z = self.rnn.prior(d).mean
            next_d = self.rnn.recurrent(z, d)
            next_zd = torch.concat([next_z, next_d], dim=1)
            y_joint = self.decoder_joint(next_zd)
            y_image = self.decoder_image(next_zd)
        return y_image, y_joint, state
