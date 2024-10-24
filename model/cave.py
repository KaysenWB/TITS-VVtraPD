

import torch
from torch import nn
from torch.nn import functional as F

class CAVE(nn.Module):
    def __init__(self, hidden_dim, latant_dim, goals):
        super(CAVE, self).__init__()
        self.hidden_dim = hidden_dim
        self.latant_dim = latant_dim
        self.goals = goals


        self.p_z_x = nn.Sequential(nn.Linear(self.hidden_dim, self.latant_dim * 4),
                                   nn.ReLU(),
                                   nn.Linear(self.latant_dim * 4, self.latant_dim * 2),
                                   nn.ReLU(),
                                   nn.Linear(self.latant_dim * 2, self.latant_dim * 2))

        self.q_z_xy = nn.Sequential(nn.Linear(self.hidden_dim + self.latant_dim * 2 , self.latant_dim * 4),
                                   nn.ReLU(),
                                   nn.Linear(self.latant_dim * 4, self.latant_dim * 2),
                                   nn.ReLU(),
                                   nn.Linear(self.latant_dim * 2, self.latant_dim * 2))


        self.pro = nn.Linear(2, self.latant_dim)
        self.goals_encoder = nn.GRU(2, self.latant_dim, bidirectional=True, batch_first=True)


    def forward(self, enc_h, cur_state, target=None ):

        z_mu_logvar_p = self.p_z_x(enc_h)
        z_mu_p = z_mu_logvar_p[:, :self.latant_dim]
        z_logvar_p = z_mu_logvar_p[:, self.latant_dim:]
        if target is not None:
            # 2. sample z from posterior, for training only
            initial_h = self.pro(cur_state)
            initial_h = torch.stack([initial_h, torch.zeros_like(initial_h, device=initial_h.device)], dim=0)
            _, target_h = self.goals_encoder(target, initial_h)
            target_h = target_h.permute(1, 0, 2)
            target_h = target_h.reshape(-1, target_h.shape[1] * target_h.shape[2])
            target_h = F.dropout(target_h, p=0.25, training=True)

            z_mu_logvar_q = self.q_z_xy(torch.cat([enc_h, target_h], dim=-1))
            z_mu_q = z_mu_logvar_q[:, :self.latant_dim]
            z_logvar_q = z_mu_logvar_q[:, self.latant_dim:]
            Z_mu = z_mu_q
            Z_logvar = z_logvar_q

            # 3. compute KL(q_z_xy||p_z_x)
            KLD = 0.5 * ((z_logvar_q.exp() / z_logvar_p.exp()) + (z_mu_p - z_mu_q).pow(2) / z_logvar_p.exp() - \
                         1 + (z_logvar_p - z_logvar_q))
            KLD = KLD.sum(dim=-1).mean()
            KLD = torch.clamp(KLD, min=0.001)
        else:
            Z_mu = z_mu_p
            Z_logvar = z_logvar_p
            KLD = 0.0

        # 4. Draw sample
        K_samples = torch.randn(enc_h.shape[0], self.goals, self.latant_dim).cuda()
        Z_std = torch.exp(0.5 * Z_logvar)
        Z = Z_mu.unsqueeze(1).repeat(1, self.goals, 1) + K_samples * Z_std.unsqueeze(1).repeat(1, self.goals, 1).cuda()

        return Z, KLD





class Decoder(nn.Module):
    def __init__(self, hidden_dim, latant_dim,pred_len, With_Z):
        super(Decoder,self).__init__()

        self.hidden_dim = hidden_dim
        self.latant_dim = latant_dim
        self.pred_len = pred_len
        self.dec_with_z = With_Z

        self.goal_decoder = nn.Sequential(nn.Linear(self.hidden_dim * 2, self.hidden_dim * 4),
                                          nn.ReLU(),
                                          nn.Linear(self.hidden_dim * 4, self.hidden_dim * 2),
                                          nn.ReLU(),
                                          nn.Linear(self.hidden_dim * 2, 2))

        self.dec_init_hidden_size = self.hidden_dim + self.latant_dim if With_Z else self.hidden_dim

        self.enc_h_to_forward_h = nn.Sequential(nn.Linear(self.dec_init_hidden_size,self.hidden_dim),
                                                nn.ReLU())
        self.traj_dec_input_forward = nn.Sequential(nn.Linear(self.hidden_dim,self.hidden_dim),
                                                    nn.ReLU())
        self.traj_dec_forward = nn.GRUCell(input_size=self.hidden_dim, hidden_size=self.hidden_dim)

        self.enc_h_to_back_h = nn.Sequential(nn.Linear(self.dec_init_hidden_size,self.hidden_dim),
                                             nn.ReLU())
        self.traj_dec_input_backward = nn.Sequential(nn.Linear(2, self.hidden_dim), # 2 or 4
                                                     nn.ReLU())
        self.traj_dec_backward = nn.GRUCell(input_size=self.hidden_dim, hidden_size=self.hidden_dim)
        self.traj_output = nn.Linear(self.hidden_dim * 2, 2) # merged forward and backward


    def forward(self,dec_in, Z):
        '''
            use a bidirectional GRU decoder to plan the path.
            Params:
                dec_h: (Nodes, hidden_dim) if not using Z in decoding, otherwise (Batch, K, dim)
                G: (Nodes, K, pred_dim)
            Returns:
                backward_outputs: (Nodes, T, K, pred_dim)
                pred_goal: (Nodes, K, 2)
            '''

        dec_in_and_z = torch.cat([dec_in.unsqueeze(1).repeat(1, Z.shape[1], 1), Z], dim=2)
        dec_h = dec_in_and_z if self.dec_with_z else dec_in

        pred_goal = self.goal_decoder(dec_in_and_z)
        K = pred_goal.shape[1]

        # 1. run forward
        forward_outputs = []
        forward_h = self.enc_h_to_forward_h(dec_h)
        if len(forward_h.shape) == 2:
            forward_h = forward_h.unsqueeze(1).repeat(1, K, 1)
        forward_h = forward_h.view(-1, forward_h.shape[-1])
        forward_input = self.traj_dec_input_forward(forward_h)
        for t in range(self.pred_len):  # the last step is the goal, no need to predict
            forward_h = self.traj_dec_forward(forward_input, forward_h)
            forward_input = self.traj_dec_input_forward(forward_h)
            forward_outputs.append(forward_h)

        forward_outputs = torch.stack(forward_outputs, dim=1)

        # 2. run backward on all samples
        backward_outputs = []
        backward_h = self.enc_h_to_back_h(dec_h)
        if len(dec_h.shape) == 2:
            backward_h = backward_h.unsqueeze(1).repeat(1, K, 1)
        backward_h = backward_h.view(-1, backward_h.shape[-1])
        backward_input = self.traj_dec_input_backward(pred_goal)  # torch.cat([G])
        backward_input = backward_input.view(-1, backward_input.shape[-1])

        for t in range(self.pred_len - 1, -1, -1):
            backward_h = self.traj_dec_backward(backward_input, backward_h)
            output = self.traj_output(torch.cat([backward_h, forward_outputs[:, t]], dim=-1))
            backward_input = self.traj_dec_input_backward(output)
            backward_outputs.append(output.view(-1, K, output.shape[-1]))

        # inverse because this is backward
        backward_outputs = backward_outputs[::-1]
        backward_outputs = torch.stack(backward_outputs, dim=1)

        return pred_goal, backward_outputs


if __name__ == '__main__':


    test_net_d = Decoder(pred_len=12,hidden_dim=32,latant_dim=32, With_Z=True)
    data_dec = torch.randn(267, 20, 64)
    data_goals = torch.randn(267, 20, 2)
    out = test_net_d(data_dec,data_goals)
    print(';')

    test_net = CAVE(hidden_dim = 320, latant_dim = 100, goals=20)
    outputs = torch.randn(19,288,320)
    y = outputs[8:,:,:].transpose(0,1)
    Z, KLD = test_net(enc_h = outputs[8,:,:], cur_state = outputs[7,:,:], target=y )

    print(';')