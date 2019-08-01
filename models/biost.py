from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from .base_model import BaseModel
from . import networks
from itertools import chain
import torch


class BiOSTModel(BaseModel):
    def name(self):
        return 'BiOSTModel'

    def compute_kl(self, mu):
        ''''
        pseudo KL loss
        taken from: https://github.com/NVlabs/MUNIT/blob/master/trainer.py
        '''
        mu_2 = torch.pow(mu, 2)
        encoding_loss = torch.mean(mu_2)
        return encoding_loss

    def set_encoders_and_decoders(self, opt):
        n_downsampling = opt.n_downsampling
        n_res_blocks = opt.n_res_blocks
        self.netEnc_a, self.netDec_a = networks.define_ED(opt.input_nc, opt.output_nc,
                                                          opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout,
                                                          opt.init_type, self.gpu_ids,
                                                          n_blocks_encoder=n_res_blocks,
                                                          n_blocks_decoder=n_res_blocks,
                                                          start=0,
                                                          end=2, n_downsampling=n_downsampling,
                                                          input_layer=True,
                                                          output_layer=True, start_dec=0,
                                                          end_dec=2)
        self.netEnc_b, self.netDec_b = networks.define_ED(opt.input_nc, opt.output_nc,
                                                          opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout,
                                                          opt.init_type, self.gpu_ids,
                                                          n_blocks_encoder=n_res_blocks,
                                                          n_blocks_decoder=n_res_blocks,
                                                          start=0,
                                                          end=2, n_downsampling=n_downsampling,
                                                          input_layer=True,
                                                          output_layer=True, start_dec=0,
                                                          end_dec=2)

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.set_encoders_and_decoders(opt)

        if self.isTrain and not opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netEnc_b, 'Enc_b', which_epoch)
            self.load_network(self.netDec_b, 'Dec_b', which_epoch)
            self.load_network(self.netEnc_a, 'Enc_b', which_epoch)
            self.load_network(self.netDec_a, 'Dec_b', which_epoch)

        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netEnc_a, 'Enc_a', which_epoch)
            self.load_network(self.netDec_a, 'Dec_a', which_epoch)
            self.load_network(self.netEnc_b, 'Enc_b', which_epoch)
            self.load_network(self.netDec_b, 'Dec_b', which_epoch)

        if self.isTrain:
             # define loss functions
            self.criterionIdt = torch.nn.L1Loss()
            self.mse = torch.nn.MSELoss()

            # initialize optimizers
            self.optimizer_a = torch.optim.Adam(chain(self.netEnc_a.parameters(), self.netDec_a.parameters()),
                                                    lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_b = torch.optim.Adam(chain(self.netEnc_b.parameters(), self.netDec_b.parameters()),
                                                    lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_a)
            self.optimizers.append(self.optimizer_b)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        if len(self.gpu_ids) > 0:
            input_A = input_A.cuda(self.gpu_ids[0], async=True)
            input_B = input_B.cuda(self.gpu_ids[0], async=True)
        self.input_A = input_A
        self.input_B = input_B
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        self.a_path = input['A_paths' if AtoB else 'B_paths']
        self.b_path = input['A_paths' if not AtoB else 'B_paths']

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_G(self):
        enc_a = self.netEnc_a(self.real_A)
        enc_b = self.netEnc_b(self.real_B)

        fake_AA = self.netDec_a(enc_a)
        fake_AB = self.netDec_b(enc_a)
        fake_BA = self.netDec_a(enc_b)
        fake_BB = self.netDec_b(enc_b)
        enc_ab = self.netEnc_b(fake_AB)
        fake_ABA = self.netDec_a(enc_ab)

        enc_ba = self.netEnc_a(fake_BA)
        fake_BAB = self.netDec_b(enc_ba)

        # Reconstruction losses
        loss_idt_A = self.opt.idt_w * self.criterionIdt(fake_AA, self.real_A)
        loss_idt_B = self.opt.idt_w * self.criterionIdt(fake_BB, self.real_B)

        # Pixel cycle losses
        loss_cycle_A = self.opt.lambda_A * self.criterionIdt(fake_ABA, self.real_A)
        loss_cycle_B = self.opt.lambda_B * self.criterionIdt(fake_BAB, self.real_B)

        # (Pseudo) KL losses
        loss_kl_B = self.opt.kl_lambda * self.compute_kl(enc_b)
        loss_kl_A = self.opt.kl_lambda * self.compute_kl(enc_a)

        # Feature cycle loss
        loss_feat_BA = self.opt.feat_weight * self.mse(enc_ba, enc_b.detach())

        loss_G_A = loss_cycle_A + loss_idt_A + loss_kl_A + loss_feat_BA
        loss_G_B = loss_cycle_B + loss_idt_B + loss_kl_B

        self.fake_AB = fake_AB.data
        self.fake_BA = fake_BA.data

        self.loss_cycle_A = loss_cycle_A.item()
        self.loss_cycle_B = loss_cycle_B.item()
        self.loss_idt_A = loss_idt_A.item()
        self.loss_idt_B = loss_idt_B.item()
        self.loss_kl_B = loss_kl_B.item()

        return loss_G_A, loss_G_B

    def optimize_parameters(self):
        # forward
        self.forward()
        loss_G_A, loss_G_B = self.backward_G()

        # x loss updates
        self.optimizer_a.zero_grad()
        loss_G_A.backward(retain_graph=True)
        self.optimizer_a.step()

        # B loss updates
        self.optimizer_b.zero_grad()
        loss_G_B.backward()
        self.optimizer_b.step()

    def get_current_errors(self):
        ret_errors = OrderedDict(
            [
             ('Idt_B', self.loss_idt_B), ('Idt_A', self.loss_idt_A),
             ('Cycle_A', self.loss_cycle_A), ('Cycle_B', self.loss_cycle_B),('Kl_B', self.loss_kl_B), ])
        return ret_errors

    def get_current_visuals(self):
        real_A = util.tensor2im(self.input_A)
        real_B = util.tensor2im(self.input_B)
        fake_AB = util.tensor2im(self.fake_AB)
        fake_BA = util.tensor2im(self.fake_BA)

        ret_visuals = OrderedDict(
            [('real_B', real_B), ('fake_BA', fake_BA),  ('fake_AB', fake_AB),
             ('real_A', real_A)
             ])
        return ret_visuals

    def save(self, label):
        self.save_network(self.netEnc_a, 'Enc_a', label, self.gpu_ids)
        self.save_network(self.netDec_a, 'Dec_a', label, self.gpu_ids)
        self.save_network(self.netEnc_b, 'Enc_b', label, self.gpu_ids)
        self.save_network(self.netDec_b, 'Dec_b', label, self.gpu_ids)
