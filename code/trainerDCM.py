from __future__ import print_function
from six.moves import range

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from PIL import Image

from miscc.config import cfg
from miscc.utils import mkdir_p
from miscc.utils import build_super_images, build_super_images2
from miscc.utils import weights_init, load_params, copy_G_params
from model import G_DCGAN, G_NET, DCM_Net
from datasets import prepare_data
from model import RNN_ENCODER, CNN_ENCODER
from VGGFeatureLoss import VGGNet

from miscc.lossesDCM import words_loss
from miscc.lossesDCM import discriminator_loss, DCM_generator_loss

import os
import time
import numpy as np
import sys

class condGANTrainer(object):
    def __init__(self, output_dir, data_loader, n_words, ixtoword):
        if cfg.TRAIN.FLAG:
            self.model_dir = os.path.join(output_dir, 'Model')
            self.image_dir = os.path.join(output_dir, 'Image')
            mkdir_p(self.model_dir)
            mkdir_p(self.image_dir)

        torch.cuda.set_device(cfg.GPU_ID)
        cudnn.benchmark = True

        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL

        self.n_words = n_words
        self.ixtoword = ixtoword
        self.data_loader = data_loader
        self.num_batches = len(self.data_loader)

    def build_models(self):
        # ################### models ######################################## #
        if cfg.TRAIN.NET_E == '':
            print('Error: no pretrained text-image encoders')
            return
        if cfg.TRAIN.NET_G == '':
            print('Error: no pretrained main module')
            return

        VGG = VGGNet()

        for p in VGG.parameters():
            p.requires_grad = False

        print("Load the VGG model")
        VGG.eval()

        image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM)
        img_encoder_path = cfg.TRAIN.NET_E.replace('text_encoder', 'image_encoder')
        state_dict = \
            torch.load(img_encoder_path, map_location=lambda storage, loc: storage)
        image_encoder.load_state_dict(state_dict)
        for p in image_encoder.parameters():
            p.requires_grad = False
        print('Load image encoder from:', img_encoder_path)
        image_encoder.eval()

        text_encoder = \
            RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
        state_dict = \
            torch.load(cfg.TRAIN.NET_E,
                       map_location=lambda storage, loc: storage)
        text_encoder.load_state_dict(state_dict)
        for p in text_encoder.parameters():
            p.requires_grad = False
        print('Load text encoder from:', cfg.TRAIN.NET_E)
        text_encoder.eval()

        if cfg.GAN.B_DCGAN:
            netG = G_DCGAN()
            from model import D_NET256 as D_NET
            netD = D_NET(b_jcu=False)
        else:
            from model import D_NET256
            netG = G_NET()
            netD = D_NET256()

        netD.apply(weights_init)

        state_dict = \
            torch.load(cfg.TRAIN.NET_G, map_location=lambda storage, loc: storage)
        netG.load_state_dict(state_dict)
        netG.eval()
        print('Load G from: ', cfg.TRAIN.NET_G)

        epoch = 0
        netDCM = DCM_Net()
        if cfg.TRAIN.NET_C != '':
            state_dict = \
                torch.load(cfg.TRAIN.NET_C, map_location=lambda storage, loc: storage)
            netDCM.load_state_dict(state_dict)
            print('Load DCM from: ', cfg.TRAIN.NET_C)
            istart = cfg.TRAIN.NET_C.rfind('_') + 1
            iend = cfg.TRAIN.NET_C.rfind('.')
            epoch = cfg.TRAIN.NET_C[istart:iend]
            epoch = int(epoch) + 1

        if cfg.TRAIN.NET_D != '':
            state_dict = \
                torch.load(cfg.TRAIN.NET_D, map_location=lambda storage, loc: storage)
            netD.load_state_dict(state_dict)
            print('Load DCM Discriminator from: ', cfg.TRAIN.NET_D)

        if cfg.CUDA:
            text_encoder = text_encoder.cuda()
            image_encoder = image_encoder.cuda()
            netG.cuda()
            netDCM.cuda()
            VGG = VGG.cuda()
            netD.cuda()
        return [text_encoder, image_encoder, netG, netD, epoch, VGG, netDCM]

    def define_optimizers(self, netDCM, netD):

        optimizerD = optim.Adam(netD.parameters(),
                             lr=cfg.TRAIN.DISCRIMINATOR_LR,
                             betas=(0.5, 0.999))

        optimizerC = optim.Adam(netDCM.parameters(),
                                lr=cfg.TRAIN.GENERATOR_LR,
                                betas=(0.5, 0.999))

        return optimizerC, optimizerD

    def prepare_labels(self):
        batch_size = self.batch_size
        real_labels = Variable(torch.FloatTensor(batch_size).fill_(1))
        fake_labels = Variable(torch.FloatTensor(batch_size).fill_(0))
        match_labels = Variable(torch.LongTensor(range(batch_size)))
        if cfg.CUDA:
            real_labels = real_labels.cuda()
            fake_labels = fake_labels.cuda()
            match_labels = match_labels.cuda()

        return real_labels, fake_labels, match_labels

    def save_model(self, netDCM, avg_param_C, netD, epoch):
        backup_para = copy_G_params(netDCM)
        load_params(netDCM, avg_param_C)
        torch.save(netDCM.state_dict(),
            '%s/netC_epoch_%d.pth' % (self.model_dir, epoch))
        load_params(netDCM, backup_para)

        torch.save(netD.state_dict(),
            '%s/netD_epoch_%d.pth' % (self.model_dir, epoch))

        print('Save C/D models.')

    def set_requires_grad_value(self, models_list, brequires):
        for i in range(len(models_list)):
            for p in models_list[i].parameters():
                p.requires_grad = brequires

    def save_img_results(self, netG, noise, sent_emb, words_embs, mask,
                         image_encoder, captions, cap_lens,
                         gen_iterations, cnn_code, region_features, 
                         real_imgs, netDCM, real_features, name='current'):
        # Save images
        fake_imgs, attention_maps, _, _, h_code, c_code = netG(noise, sent_emb,
                                        words_embs, mask, cnn_code, region_features)

        fake_img = netDCM(h_code, real_features, sent_emb, words_embs, mask, c_code)

        for i in range(len(attention_maps)):
            if len(fake_imgs) > 1:
                img = fake_imgs[i + 1].detach().cpu()
                lr_img = fake_imgs[i].detach().cpu()
            else:
                img = fake_imgs[0].detach().cpu()
                lr_img = None
            attn_maps = attention_maps[i]
            att_sze = attn_maps.size(2)
            img_set, _ = \
                build_super_images(img, captions, self.ixtoword,
                                   attn_maps, att_sze, lr_imgs=lr_img)
            if img_set is not None:
                im = Image.fromarray(img_set)
                fullpath = '%s/G_%s_%d_%d.png'\
                    % (self.image_dir, name, gen_iterations, i)
                im.save(fullpath)

        i = -1
        img = fake_imgs[i].detach()
        region_features, _ = image_encoder(img)
        att_sze = region_features.size(2)
        _, _, att_maps = words_loss(region_features.detach(),
                                    words_embs.detach(),
                                    None, cap_lens,
                                    None, self.batch_size)
        img_set, _ = \
            build_super_images(fake_imgs[i].detach().cpu(),
                               captions, self.ixtoword, att_maps, att_sze)
        if img_set is not None:
            im = Image.fromarray(img_set)
            fullpath = '%s/D_%s_%d.png'\
                % (self.image_dir, name, gen_iterations)
            im.save(fullpath)

        img_set, _ = \
            build_super_images(fake_img.detach().cpu(),
                               captions, self.ixtoword, att_maps, att_sze)

        if img_set is not None:
            im = Image.fromarray(img_set)
            fullpath = '%s/C_%s_%d.png'\
                % (self.image_dir, name, gen_iterations)
            im.save(fullpath)

        '''
        # save real images
        for k in range(8):
            im = real_imgs[2][k].data.cpu().numpy()
            im = (im + 1.0) * 127.5
            im = im.astype(np.uint8)
            im = np.transpose(im, (1, 2, 0))
            im = Image.fromarray(im)
            fullpath = '%s/R_%s_%d_%d.png'\
                    % (self.image_dir, name, gen_iterations, k)
            im.save(fullpath)
        '''

    def train(self):
        text_encoder, image_encoder, netG, netD, start_epoch, VGG, netDCM = self.build_models()
        avg_param_C = copy_G_params(netDCM)
        optimizerC, optimizerD = self.define_optimizers(netDCM, netD)
        real_labels, fake_labels, match_labels = self.prepare_labels()

        batch_size = self.batch_size
        nz = cfg.GAN.Z_DIM
        noise = Variable(torch.FloatTensor(batch_size, nz))
        fixed_noise = Variable(torch.FloatTensor(batch_size, nz).normal_(0, 1))
        if cfg.CUDA:
            noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

        gen_iterations = 0
        for epoch in range(start_epoch, self.max_epoch):
            start_t = time.time()

            data_iter = iter(self.data_loader)
            step = 0
            while step < self.num_batches:

                ######################################################
                # (1) Prepare training data and Compute text embeddings
                ######################################################
                data = data_iter.next()
                imgs, captions, cap_lens, class_ids, keys, wrong_caps, \
                                wrong_caps_len, wrong_cls_id = prepare_data(data)

                hidden = text_encoder.init_hidden(batch_size)
                # words_embs: batch_size x nef x seq_len
                # sent_emb: batch_size x nef

                # matched text embeddings
                words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
                words_embs, sent_emb = words_embs.detach(), sent_emb.detach()

                # mismatched text embeddings
                w_words_embs, w_sent_emb = text_encoder(wrong_caps, wrong_caps_len, hidden)
                w_words_embs, w_sent_emb = w_words_embs.detach(), w_sent_emb.detach()

                # image embeddings: regional and global
                region_features, cnn_code = image_encoder(imgs[cfg.TREE.BRANCH_NUM - 1])

                mask = (captions == 0)
                num_words = words_embs.size(2)
                if mask.size(1) > num_words:
                    mask = mask[:, :num_words]

                #######################################################
                # (2) Modify real images
                ######################################################
                noise.data.normal_(0, 1)
                fake_imgs, _, mu, logvar, h_code, c_code = netG(noise, sent_emb,
                                                    words_embs, mask, cnn_code, region_features)


                real_img = imgs[cfg.TREE.BRANCH_NUM - 1]
                real_features = VGG(real_img)[0]
                fake_img = netDCM(h_code, real_features, sent_emb, words_embs,\
                                         mask, c_code)

                #######################################################
                # (3) Update D network
                ######################################################
                errD = 0
                D_logs = ''

                netD.zero_grad()
                errD = discriminator_loss(netD, imgs[cfg.TREE.BRANCH_NUM - 1], 
                                        fake_img, sent_emb, real_labels, fake_labels,
                                        words_embs, cap_lens, image_encoder, class_ids, 
                                        w_words_embs, wrong_caps_len, wrong_cls_id)
                errD.backward()
                optimizerD.step()
                D_logs = 'errD: %.2f ' % (errD)

                #######################################################
                # (4) Update G network: maximize log(D(G(z)))
                ######################################################
                # compute total loss for training G
                step += 1
                gen_iterations += 1

                netDCM.zero_grad()
                errC_total, C_logs = \
                    DCM_generator_loss(netD, image_encoder, fake_img, real_labels,
                                   words_embs, sent_emb, match_labels, cap_lens,\
                                    class_ids, VGG, real_img)

                errC_total.backward()
                optimizerC.step()
                for p, avg_p in zip(netDCM.parameters(), avg_param_C):
                    avg_p.mul_(0.999).add_(0.001, p.data)

                if gen_iterations % 100 == 0:
                    print(D_logs + '\n' + C_logs)
                # save images
                if gen_iterations % 1000 == 0:
                    backup_para = copy_G_params(netDCM)
                    load_params(netDCM, avg_param_C)
                    self.save_img_results(netG, fixed_noise, sent_emb,
                                          words_embs, mask, image_encoder,
                                          captions, cap_lens, epoch, cnn_code,
                                          region_features, imgs, netDCM, real_features, name='average')
                    load_params(netDCM, backup_para)

            end_t = time.time()

            print('''[%d/%d][%d]
                  Loss_D: %.2f Loss_C: %.2f Time: %.2fs'''
                  % (epoch, self.max_epoch, self.num_batches,
                     errD, errC_total,
                     end_t - start_t))

            if epoch % cfg.TRAIN.SNAPSHOT_INTERVAL == 0:  
                self.save_model(netDCM, avg_param_C, netD, epoch)

        self.save_model(netDCM, avg_param_C, netD, self.max_epoch)

    def save_singleimages(self, images, filenames, save_dir,
                          split_dir, sentenceID=0):
        for i in range(images.size(0)):
            s_tmp = '%s/single_samples/%s/%s' %\
                (save_dir, split_dir, filenames[i])
            folder = s_tmp[:s_tmp.rfind('/')]
            if not os.path.isdir(folder):
                print('Make a new folder: ', folder)
                mkdir_p(folder)

            fullpath = '%s_%d.jpg' % (s_tmp, sentenceID)
            # range from [-1, 1] to [0, 1]
            # img = (images[i] + 1.0) / 2
            img = images[i].add(1).div(2).mul(255).clamp(0, 255).byte()
            # range from [0, 1] to [0, 255]
            ndarr = img.permute(1, 2, 0).data.cpu().numpy()
            im = Image.fromarray(ndarr)
            im.save(fullpath)

    def sampling(self, split_dir):
        if cfg.TRAIN.NET_G == '' or cfg.TRAIN.NET_C == '':
            print('Error: the path for main module or DCM is not found!')
        else:
            if split_dir == 'test':
                split_dir = 'valid'

            if cfg.GAN.B_DCGAN:
                netG = G_DCGAN()
            else:
                netG = G_NET()
            netG.apply(weights_init)
            netG.cuda()
            netG.eval()
            # The text encoder
            text_encoder = RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
            state_dict = \
                torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
            text_encoder.load_state_dict(state_dict)
            print('Load text encoder from:', cfg.TRAIN.NET_E)
            text_encoder = text_encoder.cuda()
            text_encoder.eval()
            # The image encoder
            image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM)
            img_encoder_path = cfg.TRAIN.NET_E.replace('text_encoder', 'image_encoder')
            state_dict = \
                torch.load(img_encoder_path, map_location=lambda storage, loc: storage)
            image_encoder.load_state_dict(state_dict)
            print('Load image encoder from:', img_encoder_path)
            image_encoder = image_encoder.cuda()
            image_encoder.eval()

            # The VGG network
            VGG = VGGNet()
            print("Load the VGG model")
            VGG.cuda()
            VGG.eval()

            batch_size = self.batch_size
            nz = cfg.GAN.Z_DIM
            noise = Variable(torch.FloatTensor(batch_size, nz), volatile=True)
            noise = noise.cuda()

            # The DCM
            netDCM = DCM_Net()
            if cfg.TRAIN.NET_C != '':
                state_dict = \
                    torch.load(cfg.TRAIN.NET_C, map_location=lambda storage, loc: storage)
                netDCM.load_state_dict(state_dict)
                print('Load DCM from: ', cfg.TRAIN.NET_C)
            netDCM.cuda()
            netDCM.eval()

            model_dir = cfg.TRAIN.NET_G
            state_dict = \
                torch.load(model_dir, map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load G from: ', model_dir)

            # the path to save generated images
            s_tmp = model_dir[:model_dir.rfind('.pth')]
            save_dir = '%s/%s' % (s_tmp, split_dir)
            mkdir_p(save_dir)

            cnt = 0
            idx = 0 
            for _ in range(5):  # (cfg.TEXT.CAPTIONS_PER_IMAGE):
                for step, data in enumerate(self.data_loader, 0):
                    cnt += batch_size
                    if step % 100 == 0:
                        print('step: ', step)

                    imgs, captions, cap_lens, class_ids, keys, wrong_caps, \
                                wrong_caps_len, wrong_cls_id = prepare_data(data)

                    #######################################################
                    # (1) Extract text and image embeddings
                    ######################################################

                    hidden = text_encoder.init_hidden(batch_size)

                    words_embs, sent_emb = text_encoder(wrong_caps, wrong_caps_len, hidden)
                    words_embs, sent_emb = words_embs.detach(), sent_emb.detach()

                    mask = (wrong_caps == 0)
                    num_words = words_embs.size(2)
                    if mask.size(1) > num_words:
                        mask = mask[:, :num_words]

                    region_features, cnn_code = \
                                    image_encoder(imgs[cfg.TREE.BRANCH_NUM - 1])

                    #######################################################
                    # (2) Modify real images
                    ######################################################

                    noise.data.normal_(0, 1)
                    fake_imgs, attention_maps, mu, logvar, h_code, c_code = netG(noise,
                                     sent_emb, words_embs, mask, cnn_code, region_features)
                    
                    real_img = imgs[cfg.TREE.BRANCH_NUM - 1]
                    real_features = VGG(real_img)[0]

                    fake_img = netDCM(h_code, real_features, sent_emb, words_embs,\
                                         mask, c_code)
                    for j in range(batch_size):
                        s_tmp = '%s/single' % (save_dir)
                        folder = s_tmp[:s_tmp.rfind('/')]
                        if not os.path.isdir(folder):
                            print('Make a new folder: ', folder)
                            mkdir_p(folder)
                        k = -1
                        im = fake_img[j].data.cpu().numpy()
                        im = (im + 1.0) * 127.5
                        im = im.astype(np.uint8)
                        im = np.transpose(im, (1, 2, 0))
                        im = Image.fromarray(im)
                        fullpath = '%s_s%d.png' % (s_tmp, idx)
                        idx = idx+1
                        im.save(fullpath)

    def gen_example(self, data_dic):
        if cfg.TRAIN.NET_G == '' or cfg.TRAIN.NET_C == '':
            print('Error: the path for main module or DCM is not found!')
        else:
            # The text encoder
            text_encoder = \
                RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
            state_dict = \
                torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
            text_encoder.load_state_dict(state_dict)
            print('Load text encoder from:', cfg.TRAIN.NET_E)
            text_encoder = text_encoder.cuda()
            text_encoder.eval()

            # The image encoder
            image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM)
            img_encoder_path = cfg.TRAIN.NET_E.replace('text_encoder', 'image_encoder')
            state_dict = \
                torch.load(img_encoder_path, map_location=lambda storage, loc: storage)
            image_encoder.load_state_dict(state_dict)
            print('Load image encoder from:', img_encoder_path)
            image_encoder = image_encoder.cuda()
            image_encoder.eval()

            # The VGG network
            VGG = VGGNet()
            print("Load the VGG model")
            VGG.cuda()
            VGG.eval()

            # The main module
            if cfg.GAN.B_DCGAN:
                netG = G_DCGAN()
            else:
                netG = G_NET()
            s_tmp = cfg.TRAIN.NET_G[:cfg.TRAIN.NET_G.rfind('.pth')]
            model_dir = cfg.TRAIN.NET_G
            state_dict = \
                torch.load(model_dir, map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load G from: ', model_dir)
            netG.cuda()
            netG.eval()

            # The DCM
            netDCM = DCM_Net()
            if cfg.TRAIN.NET_C != '':
                state_dict = \
                    torch.load(cfg.TRAIN.NET_C, map_location=lambda storage, loc: storage)
                netDCM.load_state_dict(state_dict)
                print('Load DCM from: ', cfg.TRAIN.NET_C)
            netDCM.cuda()
            netDCM.eval()

            for key in data_dic:
                save_dir = '%s/%s' % (s_tmp, key)
                mkdir_p(save_dir)
                captions, cap_lens, sorted_indices, imgs = data_dic[key]

                batch_size = captions.shape[0]
                nz = cfg.GAN.Z_DIM
                captions = Variable(torch.from_numpy(captions), volatile=True)
                cap_lens = Variable(torch.from_numpy(cap_lens), volatile=True)

                captions = captions.cuda()
                cap_lens = cap_lens.cuda()
                for i in range(1):  
                    noise = Variable(torch.FloatTensor(batch_size, nz), volatile=True)
                    noise = noise.cuda()

                    #######################################################
                    # (1) Extract text and image embeddings
                    ######################################################
                    hidden = text_encoder.init_hidden(batch_size)
                    
                    # The text embeddings
                    words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)

                    # The image embeddings
                    region_features, cnn_code = \
                                    image_encoder(imgs[cfg.TREE.BRANCH_NUM - 1].unsqueeze(0))
                    mask = (captions == 0)

                    #######################################################
                    # (2) Generate fake images
                    ######################################################
                    noise.data.normal_(0, 1)
                    fake_imgs, attention_maps, mu, logvar, h_code, c_code = netG(noise,
                                     sent_emb, words_embs, mask, cnn_code, region_features)
                    
                    real_img = imgs[cfg.TREE.BRANCH_NUM - 1].unsqueeze(0)
                    real_features = VGG(real_img)[0]

                    fake_img = netDCM(h_code, real_features, sent_emb, words_embs,\
                                         mask, c_code)

                    cap_lens_np = cap_lens.cpu().data.numpy()
                    for j in range(batch_size):
                        save_name = '%s/%d_s_%d' % (save_dir, i, sorted_indices[j])
                        for k in range(len(fake_imgs)):
                            im = fake_imgs[k][j].data.cpu().numpy()
                            im = (im + 1.0) * 127.5
                            im = im.astype(np.uint8)
                            im = np.transpose(im, (1, 2, 0))
                            im = Image.fromarray(im)
                            fullpath = '%s_g%d.png' % (save_name, k)
                            im.save(fullpath)

                        for k in range(len(attention_maps)):
                            if len(fake_imgs) > 1:
                                im = fake_imgs[k + 1].detach().cpu()
                            else:
                                im = fake_imgs[0].detach().cpu()
                            attn_maps = attention_maps[k]
                            att_sze = attn_maps.size(2)
                            img_set, sentences = \
                                build_super_images2(im[j].unsqueeze(0),
                                                    captions[j].unsqueeze(0),
                                                    [cap_lens_np[j]], self.ixtoword,
                                                    [attn_maps[j]], att_sze)
                            if img_set is not None:
                                im = Image.fromarray(img_set)
                                fullpath = '%s_a%d.png' % (save_name, k)
                                im.save(fullpath)


                        save_name = '%s/%d_sf_%d' % (save_dir, 1, sorted_indices[j])
                        im = fake_img[j].data.cpu().numpy()
                        im = (im + 1.0) * 127.5
                        im = im.astype(np.uint8)
                        im = np.transpose(im, (1, 2, 0))
                        im = Image.fromarray(im)
                        fullpath = '%s_SF.png' % (save_name)
                        im.save(fullpath)

                    save_name = '%s/%d_s_%d' % (save_dir, 1, 9)
                    im = imgs[2].data.cpu().numpy()
                    im = (im + 1.0) * 127.5
                    im = im.astype(np.uint8)
                    im = np.transpose(im, (1, 2, 0))
                    im = Image.fromarray(im)
                    fullpath = '%s_SR.png' % (save_name)
                    im.save(fullpath)
