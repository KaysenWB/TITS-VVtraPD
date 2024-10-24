
import torch
import torch.nn as nn
from .ts_transformer import TS_transformer
from .cave import CAVE, Decoder
from .cave_loss import cvae_loss



class VVtraPD(torch.nn.Module):

    def __init__(self, args):
        super(VVtraPD, self).__init__()

        self.args = args
        self.temporal_spatial_transformer = TS_transformer(args)

        self.full_tra_pro1 = nn.Linear(2, self.args.emsize)
        self.full_tra_pro2 = nn.Linear((self.args.seq_length  * self.args.emsize), self.args.emsize)
        self.full_tra_pro3 = nn.Linear(self.args.emsize * 2, self.args.emsize)

        self.cave_en = CAVE(hidden_dim=self.args.emsize, latant_dim = self.args.latant_dim, goals=self.args.goals)

        self.goal_decoder = nn.Sequential(nn.Linear(self.args.emsize * 2, self.args.emsize * 4),
                                          nn.ReLU(),
                                          nn.Linear(self.args.emsize * 4, self.args.emsize * 2),
                                          nn.ReLU(),
                                          nn.Linear(self.args.emsize * 2, 2))
        self.decoder = Decoder(self.args.emsize, self.args.latant_dim,self.args.pred_length, With_Z=self.args.dec_with_z)
        self.relu= nn.ReLU()
        self.dropout = nn.Dropout(0.5)


    def forward(self, inputs, iftest = False):

        nodes_abs = inputs[0]
        cur_pos = nodes_abs[(self.args.obs_length - 1), :, :]
        tar_y = nodes_abs[self.args.obs_length:, :, :].transpose(0, 1)

        # ST_transformer_embed(only observed position)
        output_ts = self.temporal_spatial_transformer(inputs=inputs) # tensor [seq, nodes, feats] [19,N,32]

        # 4.CAVE_structure
        if iftest:
            Z, KLD = self.cave_en(output_ts, cur_pos, None)
        else:
            Z, KLD = self.cave_en(output_ts, cur_pos, nodes_abs.transpose(0,1))

        # 5. decoder and predict (goals and tra)
        pred_goal, pred_traj = self.decoder(output_ts, Z)

        # 6. compute loss
        if iftest:
            loss_dict = {}
        else:
            loss_goal, loss_traj = cvae_loss(pred_goal, pred_traj, tar_y, best_of_many=self.args.best_of_many)
            loss_dict = {'loss_goal': loss_goal, 'loss_traj': loss_traj, 'loss_kld': KLD}

        return pred_goal, pred_traj, loss_dict

