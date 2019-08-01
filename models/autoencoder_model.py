import torch
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from .base_model import BaseModel
from . import networks


class AutoEncoderModel(BaseModel):
    def name(self):
        return 'AutoEncoderModel'

    def set_encoders_and_decoders(self, opt):
        n_downsampling = opt.n_downsampling
        n_res_blocks = opt.n_res_blocks
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

        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netEnc_b, 'Enc_b', which_epoch)
            self.load_network(self.netDec_b, 'Dec_b', which_epoch)

        if self.isTrain:
            # define loss functions
            self.criterionIdt = torch.nn.L1Loss()

            # initialize optimizers
            self.optimizer = torch.optim.Adam(itertools.chain(self.netEnc_b.parameters(), self.netDec_b.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netEnc_b)
        networks.print_network(self.netDec_b)
        print('-----------------------------------------------')

    def set_input(self, input):
        # 'A' is given as single_dataset
        input_B = input['A']
        if len(self.gpu_ids) > 0:
            input_B = input_B.cuda(self.gpu_ids[0], async=True)
        self.input_B = input_B
        # 'A' is given as single_dataset
        self.image_paths = input['A_paths']

    def forward(self):
        self.real_B = Variable(self.input_B)

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def compute_kl(self, mu):
        ''''
        pseudo KL loss
        taken from: https://github.com/NVlabs/MUNIT/blob/master/trainer.py
        '''
        mu_2 = torch.pow(mu, 2)
        encoding_loss = torch.mean(mu_2)
        return encoding_loss

    def backward_G(self):
        enc_b = self.netEnc_b(self.real_B)
        fake_B = self.netDec_b(enc_b)
        loss_idt_B = self.opt.lambda_B * self.criterionIdt(fake_B, self.real_B)
        loss_kl_B = self.opt.kl_lambda * self.compute_kl(enc_b)

        # combined loss
        loss_G = loss_idt_B + loss_kl_B
        loss_G.backward()

        self.fake_B = fake_B.data
        self.loss_idt_B = loss_idt_B.data[0]
        self.loss_kl_B = loss_kl_B.data[0]

    def optimize_parameters(self):
        # forward
        self.forward()

        # G
        self.optimizer.zero_grad()
        self.backward_G()
        self.optimizer.step()

    def get_current_errors(self):
        ret_errors = OrderedDict([('Idt_B', self.loss_idt_B), ('loss_kl_B', self.loss_kl_B)])
        return ret_errors

    def get_current_visuals(self):
        real_B = util.tensor2im(self.input_B)
        fake_B = util.tensor2im(self.fake_B)
        ret_visuals = OrderedDict([('real_B', real_B), ('fake_B', fake_B), ])
        return ret_visuals

    def save(self, label):
        self.save_network(self.netEnc_b, 'Enc_b', label, self.gpu_ids)
        self.save_network(self.netDec_b, 'Dec_b', label, self.gpu_ids)
