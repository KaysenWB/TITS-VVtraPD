import torch.nn as nn
from .dataloader import *
from .VVtraPD import VVtraPD
from tqdm import tqdm
from .scheduler import ParamScheduler, sigmoid_anneal
from .utils import Metrics, keep_full_tra


class processor(object):
    def __init__(self, args):

        self.args = args
        self.dataloader = Trajectory_Dataloader(args)
        self.net = VVtraPD(args)
        self.set_optimizer()
        if self.args.device == 'cuda':
            self.net.cuda()
        else:
            self.net.cpu()

        if not os.path.isdir(self.args.model_dir):
            os.mkdir(self.args.model_dir)

        self.net_file = open(os.path.join(self.args.model_dir, 'net.txt'), 'a+')
        self.net_file.write(str(self.net))
        self.net_file.close()
        self.log_file_curve = open(os.path.join(self.args.model_dir, 'log_curve.txt'), 'a+')

        self.best_ade = 100
        self.best_fde = 100
        self.best_epoch = -1

    def save_model(self, epoch):

        model_path = self.args.save_dir + '/' + self.args.train_model + '/' + self.args.train_model + '_' + \
                     str(epoch) + '.tar'
        torch.save({
            'epoch': epoch,
            'state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, model_path)

    def load_model(self):

        if self.args.load_model is not None:
            self.args.model_save_path = self.args.save_dir + '/' + self.args.train_model + '/' + self.args.train_model + '_' + \
                                        str(self.args.load_model) + '.tar'
            print(self.args.model_save_path)
            if os.path.isfile(self.args.model_save_path):
                print('Loading checkpoint')
                checkpoint = torch.load(self.args.model_save_path)
                model_epoch = checkpoint['epoch']
                self.net.load_state_dict(checkpoint['state_dict'])
                print('Loaded checkpoint at epoch', model_epoch)

    def set_optimizer(self):

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.args.learning_rate)
        self.criterion = nn.MSELoss(reduction='none')


    def test(self):

        print('Testing begin')
        self.load_model()
        self.net.eval()
        ade ,fde = self.test_epoch()
        print('Set: {}, epoch: {},ade: {} fde: {}'
        .format(self.args.test_set, self.args.load_model, ade, fde))

    def train(self):

        print('Training begin')
        test_error, test_final_error = 0, 0
        for epoch in range(self.args.num_epochs):

            self.net.train()
            train_loss = self.train_epoch(epoch)

            if epoch >= self.args.start_test:
                self.net.eval()
                ade, fde = self.test_epoch()
                if fde < self.best_fde:
                    self.best_ade = ade
                    self.best_epoch = epoch
                    self.best_fde = fde
                    self.save_model('best')

            self.log_file_curve.write(
                str(epoch) + ',' + str(train_loss) + ',' + str(test_error) + ',' + str(test_final_error) + ',' + str(
                    self.args.learning_rate) + '\n')

            if epoch % 10 == 0:
                self.log_file_curve.close()
                self.log_file_curve = open(os.path.join(self.args.model_dir, 'log_curve.txt'), 'a+')

            if epoch >= self.args.start_test:
                print(
                    '----epoch {}, train_loss={:.5f}, ADE ={:.3f}, FDE ={:.3f}, Best_ADE={:.3f}, Best_FDE={:.3f} at Epoch {}'
                    .format(epoch, train_loss, ade, fde , self.best_ade, self.best_fde,self.best_epoch))
            else:
                print('----epoch {}, train_loss={:.5f}'
                      .format(epoch, train_loss))

    def train_epoch(self, epoch):

        self.dataloader.reset_batch_pointer(set='train', valid=False)
        loss_epoch = 0

        for batch in range(self.dataloader.trainbatchnums):
            start = time.time()
            inputs, batch_id = self.dataloader.get_train_batch(batch)

            if self.args.device =='cuda':
                inputs = tuple([torch.Tensor(i).cuda() for i in inputs])
            else:
                inputs = tuple([torch.Tensor(i) for i in inputs])

            inputs = keep_full_tra(inputs)
            self.optimizer.zero_grad()

            _, _, loss_dict = self.net.forward(inputs)

            param_scheduler = ParamScheduler()
            param_scheduler.create_new_scheduler(name='kld_weight',annealer=sigmoid_anneal,
                                                annealer_kws={
                                                    'device': self.args.device,
                                                    'start': 0,
                                                    'finish': 100.0,
                                                    'center_step': 400.0,
                                                    'steps_lo_to_hi': 100.0,
                                                })
            loss = loss_dict['loss_goal'] +  loss_dict['loss_traj'] + param_scheduler.kld_weight * loss_dict['loss_kld']
            param_scheduler.step()


            loss_epoch += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.clip)
            self.optimizer.step()


            end = time.time()

            if batch % self.args.show_step == 0 and self.args.ifshow_detail:
                print('train-{}/{} (epoch {}), train_loss = {:.5f}, time/batch = {:.5f} '.format(
                    batch,self.dataloader.trainbatchnums, epoch, loss.item(), end - start))

        train_loss_epoch = loss_epoch / self.dataloader.trainbatchnums
        return train_loss_epoch

    @torch.no_grad()
    def test_epoch(self):
        self.dataloader.reset_batch_pointer(set='test')
        ADE = []
        FDE = []

        for batch in tqdm(range(self.dataloader.testbatchnums)):

            inputs, batch_id = self.dataloader.get_test_batch(batch)

            if self.args.device == 'cuda':
                inputs = tuple([torch.Tensor(i).cuda() for i in inputs])
            else:
                inputs = tuple([torch.Tensor(i) for i in inputs])

            inputs = keep_full_tra(inputs)

            _, pred_tra, _ = self.net.forward(inputs, iftest=True)

            pred_traj = pred_tra.detach().cpu().numpy()
            tar_y = inputs[0][self.args.obs_length:, :, :].transpose(0, 1).detach().cpu().numpy()
            ade, fde = Metrics(pred_traj, tar_y)

            ADE.append (ade)
            FDE.append (fde)

            self.net.zero_grad()

        return np.mean(ADE), np.mean(FDE)
