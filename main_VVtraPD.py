import argparse
import ast
import os
import pickle
import torch
import yaml
from model.processor import processor

# Use Deterministic mode and set random seed
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(0)


parser = argparse.ArgumentParser( description='VVtraPD')
parser.add_argument('--dataset', default='eth5')
parser.add_argument('--save_dir')
parser.add_argument('--model_dir')
parser.add_argument('--config')
parser.add_argument('--using_cuda', default=True, type=ast.literal_eval)
parser.add_argument('--test_set', default='eth', type=str,
                    help='Set this value to [eth, hotel, zara1, zara2, univ] for ETH-univ, ETH-hotel, UCY-zara01, UCY-zara02, UCY-univ')
parser.add_argument('--base_dir', default='.', help='Base directory including these scripts.')
parser.add_argument('--save_base_dir', default='./output/', help='Directory for saving caches and models.')
parser.add_argument('--phase', default='train', help='Set this value to \'train\' or \'test\'')
parser.add_argument('--train_model', default='VVtraPD', help='Your model name')
parser.add_argument('--load_model', default='best', type=str, help="load pretrained model for test or training")

parser.add_argument('--seq_length', default=20, type=int)
parser.add_argument('--obs_length', default=8, type=int)
parser.add_argument('--pred_length', default=12, type=int)
parser.add_argument('--batch_around_ped', default=256, type=int)
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--test_batch_size', default=4, type=int)
parser.add_argument('--show_step', default=100, type=int)
parser.add_argument('--start_test', default=0, type=int)
parser.add_argument('--num_epochs', default=170, type=int)
parser.add_argument('--ifshow_detail', default=True, type=ast.literal_eval)
parser.add_argument('--ifsave_results', default=False, type=ast.literal_eval)
parser.add_argument('--randomRotate', default=True, type=ast.literal_eval,help="=True:random rotation of each trajectory fragment")
parser.add_argument('--neighbor_thred', default=10, type=int)
parser.add_argument('--learning_rate', default=0.0015, type=float)
parser.add_argument('--clip', default=1, type=int)
parser.add_argument('--ifnoise', default=False, type=bool,help='adding noise in outputs of TS_trans module or not')
parser.add_argument('--dropout', default=0.1, type=float)

# model_TS_transformer
parser.add_argument('--emsize', default=32, type=int, help='embedding dimension')
parser.add_argument('--latant_dim', default=32, type=int, help='cave dimension and dimension of output Z')
parser.add_argument('--nhid', default=128, type=int, help='the dimension of the feedforward network model in TransformerEncoder')
parser.add_argument('--nlayers', default=1, type=int, help='the number of nn.TransformerEncoderLayer in nn.TransformerEncoder')
parser.add_argument('--nhead', default=8, type=int, help='the number of heads in the multihead-attention models')
parser.add_argument('--add_fulltra', default=True, type=bool, help='before decoder, adding full tra or not')
parser.add_argument('--dec_with_z', default=True, type=bool, help='decoder with Z or not')
parser.add_argument('--goals', default=20, type=int)
parser.add_argument('--best_of_many', default=True, type=bool, help='whether use best loss in cave_loss or not')
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--device_id', default=0, type=int)


args = parser.parse_args()

args.save_dir = args.save_base_dir + str(args.test_set) + '/'
args.model_dir = args.save_base_dir + str(args.test_set) + '/' + args.train_model + '/'
if not os.path.exists(args.model_dir):
    os.makedirs(args.model_dir)

if args.device =='cuda':
    torch.cuda.set_device(args.device_id)

trainer = processor(args)
trainer.train()
trainer.test()