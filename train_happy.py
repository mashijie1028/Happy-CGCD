import argparse
import os
import math
import time
from tqdm import tqdm
from copy import deepcopy

from sklearn.cluster import KMeans
import numpy as np
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.optim import SGD, lr_scheduler

from project_utils.general_utils import set_seed, init_experiment, AverageMeter
from project_utils.cluster_and_log_utils import log_accs_from_preds

from data.augmentations import get_transform
from data.get_datasets import get_class_splits, ContrastiveLearningViewGenerator, get_datasets

from models.utils_simgcd import DINOHead, get_params_groups, SupConLoss, info_nce_logits, DistillLoss
from models.utils_simgcd_pro import get_kmeans_centroid_for_new_head
from models.utils_proto_aug import ProtoAugManager
from models import vision_transformer as vits
from config import dino_pretrain_path, exp_root_happy

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


'''offline train and test'''
'''====================================================================================================================='''
def train_offline(student, train_loader, test_loader, args):

    params_groups = get_params_groups(student)
    optimizer = SGD(params_groups, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs_offline,
            eta_min=args.lr * 1e-3,
        )

    cluster_criterion = DistillLoss(
                        args.warmup_teacher_temp_epochs,
                        args.epochs_offline,
                        args.n_views,
                        args.warmup_teacher_temp,
                        args.teacher_temp,
                    )
    # best acc log
    best_test_acc_old = 0

    for epoch in range(args.epochs_offline):
        loss_record = AverageMeter()

        student.train()
        for batch_idx, batch in enumerate(train_loader):

            images, class_labels, uq_idxs = batch   # NOTE!!! no mask lab in this setting
            mask_lab = torch.ones_like(class_labels)   # NOTE!!! all samples are labeled

            class_labels, mask_lab = class_labels.cuda(non_blocking=True), mask_lab.cuda(non_blocking=True).bool()
            images = torch.cat(images, dim=0).cuda(non_blocking=True)

            student_proj, student_out = student(images)
            teacher_out = student_out.detach()

            # clustering, sup
            sup_logits = torch.cat([f[mask_lab] for f in (student_out / 0.1).chunk(2)], dim=0)
            sup_labels = torch.cat([class_labels[mask_lab] for _ in range(2)], dim=0)
            cls_loss = nn.CrossEntropyLoss()(sup_logits, sup_labels)

            # clustering, unsup
            cluster_loss = cluster_criterion(student_out, teacher_out, epoch)
            avg_probs = (student_out / 0.1).softmax(dim=1).mean(dim=0)
            me_max_loss = - torch.sum(torch.log(avg_probs**(-avg_probs))) + math.log(float(len(avg_probs)))
            cluster_loss += args.memax_weight * me_max_loss

            # represent learning, unsup
            contrastive_logits, contrastive_labels = info_nce_logits(features=student_proj)
            contrastive_loss = torch.nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)

            # representation learning, sup
            student_proj = torch.cat([f[mask_lab].unsqueeze(1) for f in student_proj.chunk(2)], dim=1)
            student_proj = torch.nn.functional.normalize(student_proj, dim=-1)
            sup_con_labels = class_labels[mask_lab]
            sup_con_loss = SupConLoss()(student_proj, labels=sup_con_labels)

            # Total loss
            loss = 0
            loss += (1 - args.sup_weight) * cluster_loss + args.sup_weight * cls_loss
            loss += (1 - args.sup_weight) * contrastive_loss + args.sup_weight * sup_con_loss

            # logs
            pstr = ''
            pstr += f'cls_loss: {cls_loss.item():.4f} '
            pstr += f'cluster_loss: {cluster_loss.item():.4f} '
            pstr += f'sup_con_loss: {sup_con_loss.item():.4f} '
            pstr += f'contrastive_loss: {contrastive_loss.item():.4f} '

            loss_record.update(loss.item(), class_labels.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % args.print_freq == 0:
                args.logger.info('Epoch: [{}][{}/{}]\t loss {:.5f}\t {}'
                            .format(epoch, batch_idx, len(train_loader), loss.item(), pstr))

        args.logger.info('Train Epoch: {} Avg Loss: {:.4f} '.format(epoch, loss_record.avg))

        args.logger.info('Testing on disjoint test set...')
        all_acc_test, old_acc_test, _ = test_offline(student, test_loader, epoch=epoch, save_name='Test ACC', args=args)
        args.logger.info('Test Accuracies: All {:.4f} | Old {:.4f}'.format(all_acc_test, old_acc_test))

        # Step schedule
        exp_lr_scheduler.step()

        save_dict = {
            'model': student.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
        }

        torch.save(save_dict, args.model_path)
        args.logger.info("model saved to {}.".format(args.model_path))

        if old_acc_test > best_test_acc_old:

            args.logger.info(f'Best ACC on Old Classes on test set: {old_acc_test:.4f}...')

            torch.save(save_dict, args.model_path[:-3] + f'_best.pt')
            args.logger.info("model saved to {}.".format(args.model_path[:-3] + f'_best.pt'))

            best_test_acc_old = old_acc_test

        args.logger.info(f'Exp Name: {args.exp_name}')
        args.logger.info(f'Metrics with best model on test set: Old: {best_test_acc_old:.4f}')
        args.logger.info('\n')


def test_offline(model, test_loader, epoch, save_name, args):

    model.eval()

    preds, targets = [], []
    mask = np.array([])
    # First extract all features
    for batch_idx, (images, label, _) in enumerate(tqdm(test_loader)):
        images = images.cuda(non_blocking=True)
        with torch.no_grad():
            _, logits = model(images)
            preds.append(logits.argmax(1).cpu().numpy())
            targets.append(label.cpu().numpy())
            mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes)) else False for x in label]))

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    # -----------------------
    # EVALUATE
    # -----------------------
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    args=args)

    return all_acc, old_acc, new_acc
'''====================================================================================================================='''




'''online train and test'''
'''====================================================================================================================='''
def train_online(student, student_pre, proto_aug_manager, train_loader, test_loader, current_session, args):

    params_groups = get_params_groups(student)
    optimizer = SGD(params_groups, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs_online_per_session,
            eta_min=args.lr * 1e-3,
        )

    cluster_criterion = DistillLoss(
                        args.warmup_teacher_temp_epochs,
                        args.epochs_online_per_session,
                        args.n_views,
                        args.warmup_teacher_temp,
                        args.teacher_temp,
                    )

    # best acc log
    best_test_acc_all = 0
    best_test_acc_old = 0
    best_test_acc_new = 0

    best_test_acc_soft_all = 0
    best_test_acc_seen = 0
    best_test_acc_unseen = 0

    for epoch in range(args.epochs_online_per_session):
        loss_record = AverageMeter()

        student.train()
        student_pre.eval()
        for batch_idx, batch in enumerate(train_loader):

            images, class_labels, uq_idxs, _ = batch   # NOTE!!!   mask lab in this setting
            mask_lab = torch.zeros_like(class_labels)   # NOTE!!! all samples are unlabeled

            class_labels, mask_lab = class_labels.cuda(non_blocking=True), mask_lab.cuda(non_blocking=True).bool()
            images = torch.cat(images, dim=0).cuda(non_blocking=True)

            student_proj, student_out = student(images)
            teacher_out = student_out.detach()

            # clustering, unsup
            cluster_loss = cluster_criterion(student_out, teacher_out, epoch)
            avg_probs = (student_out / 0.1).softmax(dim=1).mean(dim=0)
            #me_max_loss = - torch.sum(torch.log(avg_probs**(-avg_probs))) + math.log(float(len(avg_probs)))
            #cluster_loss += args.memax_weight * me_max_loss

            # 1. inter old and new
            avg_probs_old_in = avg_probs[:args.num_seen_classes]
            avg_probs_new_in = avg_probs[args.num_seen_classes:]

            #avg_probs_old_new = torch.tensor([torch.sum(avg_probs_old_in), torch.sum(avg_probs_new_in)], requires_grad=True, device=device)
            #me_max_loss_old_new = - torch.sum(torch.log(avg_probs_old_new**(-avg_probs_old_new))) + math.log(float(len(avg_probs_old_new)))
            avg_probs_old_marginal, avg_probs_new_marginal = torch.sum(avg_probs_old_in), torch.sum(avg_probs_new_in)
            me_max_loss_old_new =  avg_probs_old_marginal * torch.log(avg_probs_old_marginal) + avg_probs_new_marginal * torch.log(avg_probs_new_marginal) + math.log(2)

            # 2. old (intra) & new (intra)
            avg_probs_old_in_norm = avg_probs_old_in / torch.sum(avg_probs_old_in)   # norm
            avg_probs_new_in_norm = avg_probs_new_in / torch.sum(avg_probs_new_in)   # norm
            me_max_loss_old_in = - torch.sum(torch.log(avg_probs_old_in_norm**(-avg_probs_old_in_norm))) + math.log(float(len(avg_probs_old_in_norm)))
            if args.num_novel_class_per_session > 1:
                me_max_loss_new_in = - torch.sum(torch.log(avg_probs_new_in_norm**(-avg_probs_new_in_norm))) + math.log(float(len(avg_probs_new_in_norm)))
            else:
                me_max_loss_new_in = torch.tensor(0.0, device=device)
            # overall me-max loss
            cluster_loss += args.memax_old_new_weight * me_max_loss_old_new + \
                args.memax_old_in_weight * me_max_loss_old_in + args.memax_new_in_weight * me_max_loss_new_in


            # represent learning, unsup
            contrastive_logits, contrastive_labels = info_nce_logits(features=student_proj)
            contrastive_loss = torch.nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)

            proto_aug_loss = proto_aug_manager.compute_proto_aug_hardness_aware_loss(student)
            feats = student[0](images)
            feats = torch.nn.functional.normalize(feats, dim=-1)
            with torch.no_grad():
                feats_pre = student_pre[0](images)
                feats_pre = torch.nn.functional.normalize(feats_pre, dim=-1)
            feat_distill_loss = (feats-feats_pre).pow(2).sum() / len(feats)

            # Total loss
            loss = 0
            loss += 1 * cluster_loss
            loss += 1 * contrastive_loss
            loss += args.proto_aug_weight * proto_aug_loss
            loss += args.feat_distill_weight * feat_distill_loss

            # logs
            pstr = ''
            pstr += f'me_max_loss_old_new: {me_max_loss_old_new.item():.4f} '
            pstr += f'me_max_loss_old_in: {me_max_loss_old_in.item():.4f} '
            pstr += f'me_max_loss_new_in: {me_max_loss_new_in.item():.4f} '
            pstr += f'cluster_loss: {cluster_loss.item():.4f} '
            pstr += f'contrastive_loss: {contrastive_loss.item():.4f} '
            pstr += f'proto_aug_loss: {proto_aug_loss.item():.4f} '
            pstr += f'feat_distill_loss: {feat_distill_loss.item():.4f} '

            loss_record.update(loss.item(), class_labels.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % args.print_freq == 0:
                args.logger.info('Epoch: [{}][{}/{}]\t loss {:.5f}\t {}'
                            .format(epoch, batch_idx, len(train_loader), loss.item(), pstr))
                new_true_ratio = len(class_labels[class_labels>=args.num_seen_classes]) / len(class_labels)
                logits = student_out / 0.1
                preds = logits.argmax(1)
                new_pred_ratio = len(preds[preds>=args.num_seen_classes]) / len(preds)
                args.logger.info(f'Avg old prob: {torch.sum(avg_probs_old_in).item():.4f} | Avg new prob: {torch.sum(avg_probs_new_in).item():.4f} | Pred new ratio: {new_pred_ratio:.4f} | Ground-truth new ratio: {new_true_ratio:.4f}')

        args.logger.info('Train Epoch: {} Avg Loss: {:.4f} '.format(epoch, loss_record.avg))

        args.logger.info('Testing on disjoint test set...')
        all_acc_test, old_acc_test, new_acc_test, \
            all_acc_soft_test, seen_acc_test, unseen_acc_test = test_online(student, test_loader, epoch=epoch, save_name='Test ACC', args=args)
        args.logger.info('Test Accuracies (Hard): All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc_test, old_acc_test, new_acc_test))
        args.logger.info('Test Accuracies (Soft): All {:.4f} | Seen {:.4f} | Unseen {:.4f}'.format(all_acc_soft_test, seen_acc_test, unseen_acc_test))

        # Step schedule
        exp_lr_scheduler.step()

        save_dict = {
            'model': student.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
        }

        #torch.save(save_dict, args.model_path[:-3] + '_session-' + str(current_session) + f'.pt')   # NOTE!!! session
        #args.logger.info("model saved to {}.".format(args.model_path[:-3] + '_session-' + str(current_session) + f'.pt'))

        if all_acc_test > best_test_acc_all:

            args.logger.info(f'Best ACC on All Classes on test set of session-{current_session}: {all_acc_test:.4f}...')

            torch.save(save_dict, args.model_path[:-3] + '_session-' + str(current_session) + f'_best.pt')   # NOTE!!! session
            args.logger.info("model saved to {}.".format(args.model_path[:-3] + '_session-' + str(current_session) + f'_best.pt'))

            best_test_acc_all = all_acc_test
            best_test_acc_old = old_acc_test
            best_test_acc_new = new_acc_test

            best_test_acc_soft_all = all_acc_soft_test
            best_test_acc_seen = seen_acc_test
            best_test_acc_unseen = unseen_acc_test

        args.logger.info(f'Exp Name: {args.exp_name}')
        args.logger.info(f'Metrics with best model on test set (Hard) of session-{current_session}: All (Hard): {best_test_acc_all:.4f} Old: {best_test_acc_old:.4f} New: {best_test_acc_new:.4f}')
        args.logger.info(f'Metrics with best model on test set (Hard) of session-{current_session}: All (Soft): {best_test_acc_soft_all:.4f} Seen: {best_test_acc_seen:.4f} Unseen: {best_test_acc_unseen:.4f}')
        args.logger.info('\n')


    # log best test acc list
    args.best_test_acc_all_list.append(best_test_acc_all)
    args.best_test_acc_old_list.append(best_test_acc_old)
    args.best_test_acc_new_list.append(best_test_acc_new)
    args.best_test_acc_soft_all_list.append(best_test_acc_soft_all)
    args.best_test_acc_seen_list.append(best_test_acc_seen)
    args.best_test_acc_unseen_list.append(best_test_acc_unseen)



def test_online(model, test_loader, epoch, save_name, args):

    model.eval()

    preds, targets = [], []
    mask_hard = np.array([])
    mask_soft = np.array([])
    for batch_idx, (images, label, _) in enumerate(tqdm(test_loader)):
        images = images.cuda(non_blocking=True)
        with torch.no_grad():
            _, logits = model(images)
            preds.append(logits.argmax(1).cpu().numpy())
            targets.append(label.cpu().numpy())
            mask_hard = np.append(mask_hard, np.array([True if x.item() in range(len(args.train_classes))
                                         else False for x in label]))
            mask_soft = np.append(mask_soft, np.array([True if x.item() in range(args.num_seen_classes)
                                         else False for x in label]))

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    # -----------------------
    # EVALUATE
    # -----------------------
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask_hard,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    args=args)

    all_acc_soft, seen_acc, unseen_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask_soft,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    args=args)

    return all_acc, old_acc, new_acc, all_acc_soft, seen_acc, unseen_acc
'''====================================================================================================================='''


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='cluster', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--num_workers_test', default=4, type=int)
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default='v2')

    parser.add_argument('--dataset_name', type=str, default='cifar100', help='options: cifar10, cifar100, tiny_imagenet, cub, imagenet_100')
    parser.add_argument('--use_ssb_splits', action='store_true', default=True)

    parser.add_argument('--grad_from_block', type=int, default=11)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--exp_root', type=str, default=exp_root_happy)
    parser.add_argument('--transform', type=str, default='imagenet')

    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--sup_weight', type=float, default=0.35)
    parser.add_argument('--n_views', default=2, type=int)
    parser.add_argument('--contrast_unlabel_only', action='store_true', default=False)

    '''group-wise entropy regularization'''
    # memax weight for offline session
    parser.add_argument('--memax_weight', type=float, default=1)
    # memax weight for online session
    parser.add_argument('--memax_old_new_weight', type=float, default=2)
    parser.add_argument('--memax_old_in_weight', type=float, default=1)
    parser.add_argument('--memax_new_in_weight', type=float, default=1)
    parser.add_argument('--warmup_teacher_temp', default=0.07, type=float, help='Initial value for the teacher temperature.')
    parser.add_argument('--teacher_temp', default=0.04, type=float, help='Final value (after linear warmup) of the teacher temperature.')
    #parser.add_argument('--teacher_temp_final', default=0.05, type=float, help='Final value (online session) of the teacher temperature.')
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int, help='Number of warmup epochs for the teacher temperature.')

    '''clustering-guided initialization'''
    parser.add_argument('--init_new_head', action='store_true', default=False)

    '''PASS params'''
    parser.add_argument('--proto_aug_weight', type=float, default=1.0)
    parser.add_argument('--feat_distill_weight', type=float, default=1.0)
    parser.add_argument('--radius_scale', type=float, default=1.0)

    '''hardness-aware sampling temperature'''
    parser.add_argument('--hardness_temp', type=float, default=0.1)

    # Continual GCD params
    parser.add_argument('--num_old_classes', type=int, default=-1)
    parser.add_argument('--prop_train_labels', type=float, default=0.8)
    parser.add_argument('--train_session', type=str, default='offline', help='options: offline, online')
    parser.add_argument('--load_offline_id', type=str, default=None)
    parser.add_argument('--epochs_offline', default=100, type=int)
    parser.add_argument('--epochs_online_per_session', default=30, type=int)
    parser.add_argument('--continual_session_num', default=4, type=int)
    parser.add_argument('--online_novel_unseen_num', default=400, type=int)
    parser.add_argument('--online_old_seen_num', default=50, type=int)
    parser.add_argument('--online_novel_seen_num', default=50, type=int)

    # shuffle dataset classes
    parser.add_argument('--shuffle_classes', action='store_true', default=False)
    parser.add_argument('--seed', default=0, type=int)

    # others
    parser.add_argument('--print_freq', default=10, type=int)
    parser.add_argument('--exp_name', default='simgcd-pro-v5', type=str)


    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    device = torch.device('cuda:0')
    #set_seed(args.seed)
    args = get_class_splits(args)

    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    args.exp_root = args.exp_root + '_' + args.train_session
    args.exp_name = 'happy' + '-' + args.train_session

    if args.train_session == 'offline':
        args.base_exp_id = 'Old' + str(args.num_labeled_classes) + '_' + 'Ratio' + str(args.prop_train_labels)

    elif args.train_session == 'online':
        args.base_exp_id = 'Old' + str(args.num_labeled_classes) + '_' + 'Ratio' + str(args.prop_train_labels) \
            + '_' + 'ContinualNum' + str(args.continual_session_num) + '_' + 'UnseenNum' + str(args.online_novel_unseen_num) \
                + '_' + 'SeenNum' + str(args.online_novel_seen_num)

    else:
        raise NotImplementedError

    init_experiment(args, runner_name=['Happy'])
    args.logger.info(f'Using evaluation function {args.eval_funcs[0]} to print results')

    # ----------------------
    # BASE MODEL
    # ----------------------
    args.interpolation = 3
    args.crop_pct = 0.875

    backbone = vits.__dict__['vit_base']()

    args.logger.info(f'Loading weights from {dino_pretrain_path}')
    state_dict = torch.load(dino_pretrain_path, map_location='cpu')
    backbone.load_state_dict(state_dict)

    # NOTE: Hardcoded image size as we do not finetune the entire ViT model
    args.image_size = 224
    args.feat_dim = 768
    args.num_mlp_layers = 3
    args.mlp_out_dim = args.num_labeled_classes   # NOTE!!!

    # ----------------------
    # HOW MUCH OF BASE MODEL TO FINETUNE
    # ----------------------
    for m in backbone.parameters():
        m.requires_grad = False

    # Only finetune layers from block 'args.grad_from_block' onwards
    for name, m in backbone.named_parameters():
        if 'block' in name:
            block_num = int(name.split('.')[1])
            if block_num >= args.grad_from_block:
                m.requires_grad = True

    args.logger.info('model build')

    # ----------------------
    # PROJECTION HEAD
    # ----------------------
    projector = DINOHead(in_dim=args.feat_dim, out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers)
    model = nn.Sequential(backbone, projector)

    model.to(device)

    # --------------------
    # CONTRASTIVE TRANSFORM
    # --------------------
    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)


    # ----------------------
    # 1. OFFLINE TRAIN
    # ----------------------
    if args.train_session == 'offline':
        args.logger.info('========== offline training with labeled old data (old) ==========')
        args.logger.info('loading dataset...')
        offline_session_train_dataset, offline_session_test_dataset,\
            _online_session_train_dataset_list, _online_session_test_dataset_list,\
                datasets, dataset_split_config_dict, novel_targets_shuffle = get_datasets(
                    args.dataset_name, train_transform, test_transform, args)

        # saving dataset dict
        print('save dataset dict...')
        save_dataset_dict_path = os.path.join(args.log_dir, 'offline_dataset_dict.txt')
        f_dataset_dict = open(save_dataset_dict_path, 'w')
        f_dataset_dict.write('offline_dataset_split_dict: \n')
        f_dataset_dict.write(str(dataset_split_config_dict))
        f_dataset_dict.write('\nnovel_targets_shuffle: \n')
        f_dataset_dict.write(str(novel_targets_shuffle))
        f_dataset_dict.close()
        
        offline_session_train_loader = DataLoader(offline_session_train_dataset, num_workers=args.num_workers,
                                                  batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=True)
        offline_session_test_loader = DataLoader(offline_session_test_dataset, num_workers=args.num_workers_test,
                                                 batch_size=256, shuffle=False, pin_memory=False)

        # ----------------------
        # TRAIN
        # ----------------------
        train_offline(model, offline_session_train_loader, offline_session_test_loader, args)


    # ----------------------
    # 2. ONLINE TRAIN
    # ----------------------
    elif args.train_session == 'online':
        args.logger.info('\n\n==================== online continual GCD with unlabeled data (old + novel) ====================')
        args.logger.info('loading dataset...')
        _offline_session_train_dataset, _offline_session_test_dataset,\
            online_session_train_dataset_list, online_session_test_dataset_list,\
                datasets, dataset_split_config_dict, novel_targets_shuffle = get_datasets(
                    args.dataset_name, train_transform, test_transform, args)

        # saving dataset dict
        print('save dataset dict...')
        save_dataset_dict_path = os.path.join(args.log_dir, 'online_dataset_dict.txt')
        f_dataset_dict = open(save_dataset_dict_path, 'w')
        f_dataset_dict.write('online_dataset_split_dict: \n')
        f_dataset_dict.write(str(dataset_split_config_dict))
        f_dataset_dict.write('\nnovel_targets_shuffle: \n')
        f_dataset_dict.write(str(novel_targets_shuffle))
        f_dataset_dict.write('\nnum_novel_class_per_session: \n')
        f_dataset_dict.write(str(args.num_unlabeled_classes // args.continual_session_num))
        f_dataset_dict.close()


        # ----------------------
        # CONTINUAL SESSIONS
        # ----------------------
        args.num_novel_class_per_session = args.num_unlabeled_classes // args.continual_session_num
        args.logger.info('number of novel class per session: {}'.format(args.num_novel_class_per_session))

        '''v5: ProtoAug Manager'''
        proto_aug_manager = ProtoAugManager(args.feat_dim, args.n_views*args.batch_size, args.hardness_temp, args.radius_scale, device, args.logger)

        # best test acc list across continual sessions
        args.best_test_acc_all_list = []
        args.best_test_acc_old_list = []
        args.best_test_acc_new_list = []
        args.best_test_acc_soft_all_list = []
        args.best_test_acc_seen_list = []
        args.best_test_acc_unseen_list = []

        start_session = 0

        '''Continual GCD sessions'''
        #for session in range(args.continual_session_num):
        for session in range(start_session, args.continual_session_num):
            args.logger.info('\n\n========== begin online continual session-{} ==============='.format(session+1))
            # dataset for the current session
            online_session_train_dataset = online_session_train_dataset_list[session]
            online_session_test_dataset = online_session_test_dataset_list[session]

            online_session_train_loader = DataLoader(online_session_train_dataset, num_workers=args.num_workers,
                                                     batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=True)
            online_session_test_loader = DataLoader(online_session_test_dataset, num_workers=args.num_workers_test,
                                                    batch_size=256, shuffle=False, pin_memory=False)

            # number of seen (offline old + previous online new) classes till the beginning of this session
            args.num_seen_classes = args.num_labeled_classes + args.num_novel_class_per_session * session
            args.logger.info('number of seen class (old + seen novel) at the beginning of current session: {}'.format(args.num_seen_classes))
            if args.dataset_name == 'cifar100':
                args.num_cur_novel_classes = len(np.unique(online_session_train_dataset.novel_unlabelled_dataset.targets))
            elif args.dataset_name == 'tiny_imagenet':
                novel_cls_labels = [t for i, (p, t) in enumerate(online_session_train_dataset.novel_unlabelled_dataset.data)]
                args.num_cur_novel_classes = len(np.unique(novel_cls_labels))
            elif args.dataset_name == 'aircraft':
                novel_cls_labels = [t for i, (p, t) in enumerate(online_session_train_dataset.novel_unlabelled_dataset.samples)]
                args.num_cur_novel_classes = len(np.unique(novel_cls_labels))
            elif args.dataset_name == 'scars':
                args.num_cur_novel_classes = len(np.unique(online_session_train_dataset.novel_unlabelled_dataset.target))   # NOTE!!! target
            else:
                args.num_cur_novel_classes = args.num_novel_class_per_session * (session+1)
            args.logger.info('number of all novel class (seen novel + unseen novel) in current session: {}'.format(args.num_cur_novel_classes))


            '''tunable params in backbone'''
            ####################################################################################################################
            # freeze backbone params
            for m in backbone.parameters():
                m.requires_grad = False

            # Only finetune layers from block 'args.grad_from_block' onwards
            for name, m in backbone.named_parameters():
                if 'block' in name:
                    block_num = int(name.split('.')[1])
                    if block_num >= args.grad_from_block:
                        m.requires_grad = True
            ####################################################################################################################

            '''load ckpts from last session (session>0) or offline session (session=0)'''
            ####################################################################################################################
            args.logger.info('loading checkpoints of model_pre...')
            if session == 0:
                projector_pre = DINOHead(in_dim=args.feat_dim, out_dim=args.num_labeled_classes, nlayers=args.num_mlp_layers)
                model_pre = nn.Sequential(backbone, projector_pre)
                if args.load_offline_id is not None:
                    load_dir_online = os.path.join(exp_root_happy + '_' + 'offline', args.dataset_name, args.load_offline_id, 'checkpoints', 'model_best.pt')
                    args.logger.info('loading offline checkpoints from: ' + load_dir_online)
                    load_dict = torch.load(load_dir_online)
                    model_pre.load_state_dict(load_dict['model'])
                    args.logger.info('successfully loaded checkpoints!')
            else:        # session > 0:
                projector_pre = DINOHead(in_dim=args.feat_dim, out_dim=args.num_seen_classes, nlayers=args.num_mlp_layers)
                model_pre = nn.Sequential(backbone, projector_pre)
                load_dir_online = args.model_path[:-3] + '_session-' + str(session) + f'_best.pt'   # NOTE!!! session, best
                args.logger.info('loading checkpoints from last online session: ' + load_dir_online)
                load_dict = torch.load(load_dir_online)
                model_pre.load_state_dict(load_dict['model'])
                args.logger.info('successfully loaded checkpoints!')
            ####################################################################################################################

            '''incremental parametric classifier in SimGCD'''
            ####################################################################################################################
            ####################################################################################################################
            backbone_cur = deepcopy(backbone)   # NOTE!!!
            backbone_cur.load_state_dict(model_pre[0].state_dict())   # NOTE!!!
            args.mlp_out_dim_cur = args.num_labeled_classes + args.num_cur_novel_classes   # total num of classes in the current session
            args.logger.info('number of all class (old + all new) in current session: {}'.format(args.mlp_out_dim_cur))
            projector_cur = DINOHead(in_dim=args.feat_dim, out_dim=args.mlp_out_dim_cur, nlayers=args.num_mlp_layers)
            args.logger.info('transferring classification head of seen classes...')
            projector_cur.last_layer.weight_v.data[:args.num_seen_classes] = projector_pre.last_layer.weight_v.data[:args.num_seen_classes]   # NOTE!!!
            projector_cur.last_layer.weight_g.data[:args.num_seen_classes] = projector_pre.last_layer.weight_g.data[:args.num_seen_classes]   # NOTE!!!
            projector_cur.last_layer.weight.data[:args.num_seen_classes] = projector_pre.last_layer.weight.data[:args.num_seen_classes]   # NOTE!!!
            # initialize new class heads
            #############################################
            online_session_train_dataset_for_new_head_init = deepcopy(online_session_train_dataset)
            online_session_train_dataset_for_new_head_init.old_unlabelled_dataset.transform = test_transform   # NOTE!!!
            online_session_train_dataset_for_new_head_init.novel_unlabelled_dataset.transform = test_transform   # NOTE!!!
            online_session_train_loader_for_new_head_init = DataLoader(online_session_train_dataset_for_new_head_init, num_workers=args.num_workers_test,
                                                                    batch_size=256, shuffle=False, pin_memory=False)
            if args.init_new_head:
                new_head = get_kmeans_centroid_for_new_head(model_pre, online_session_train_loader_for_new_head_init, args, device)   # torch.Size([10, 768])
                norm_new_head_weight_v = torch.norm(projector_cur.last_layer.weight_v.data[args.num_seen_classes:], dim=-1).mean()
                norm_new_head_weight = torch.norm(projector_cur.last_layer.weight.data[args.num_seen_classes:], dim=-1).mean()
                new_head_weight_v = new_head * norm_new_head_weight_v
                new_head_weight = new_head * norm_new_head_weight
                args.logger.info('initializing classification head of unseen novel classes...')
                projector_cur.last_layer.weight_v.data[args.num_seen_classes:] = new_head_weight_v.data   # NOTE!!!   # copy
                projector_cur.last_layer.weight.data[args.num_seen_classes:] = new_head_weight.data   # NOTE!!!
            ##############################################

            model_cur = nn.Sequential(backbone_cur, projector_cur)   # NOTE!!! backbone_cur
            args.logger.info('incremental classifier heads from {} to {}'.format(len(model_pre[1].last_layer.weight_v), len(model_cur[1].last_layer.weight_v)))
            model_cur.to(device)
            ####################################################################################################################
            ####################################################################################################################

            '''compute prototypes offline (session = 0)'''
            if session == 0:
                args.logger.info('Before Train: compute offline prototypes and radius from {} classes with the best model...'.format(args.num_labeled_classes))
                offline_session_train_dataset_for_proto_aug = deepcopy(_offline_session_train_dataset)
                offline_session_train_dataset_for_proto_aug.transform = test_transform
                offline_session_train_loader_for_proto_aug = DataLoader(offline_session_train_dataset_for_proto_aug, num_workers=args.num_workers_test,
                                                                        batch_size=256, shuffle=False, pin_memory=False)
                # NOTE!!! use model_pre && offline_session_train_loader
                proto_aug_manager.update_prototypes_offline(model_pre, offline_session_train_loader_for_proto_aug, args.num_labeled_classes)
                save_path = os.path.join(args.model_dir, 'ProtoAugDict' + '_offline' + f'.pt')
                args.logger.info('Saving ProtoAugDict to {}.'.format(save_path))
                proto_aug_manager.save_proto_aug_dict(save_path)

            # ----------------------
            # TRAIN
            # ----------------------
            train_online(model_cur, model_pre, proto_aug_manager, online_session_train_loader, online_session_test_loader, session+1, args)

            '''compute prototypes online after train (session > 0)'''
            #############################################################################################################
            args.logger.info('After Train: update online prototypes from {} to {} classes with the best model...'.format(args.num_seen_classes, args.num_labeled_classes + args.num_cur_novel_classes))
            # NOTE!!! use model_cur_best && online_session_train_loader
            load_dir_online_best = args.model_path[:-3] + '_session-' + str(session+1) + f'_best.pt'   # NOTE!!! session, best
            args.logger.info('loading best checkpoints current online session: ' + load_dir_online_best)
            load_dict = torch.load(load_dir_online_best)
            model_cur.load_state_dict(load_dict['model'])
            proto_aug_manager.update_prototypes_online(model_cur, online_session_train_loader_for_new_head_init, 
                                                       args.num_seen_classes, args.num_labeled_classes + args.num_cur_novel_classes)
            save_path = os.path.join(args.model_dir, 'ProtoAugDict' + '_session-' + str(session+1) + f'.pt')
            args.logger.info('Saving ProtoAugDict to {}.'.format(save_path))
            proto_aug_manager.save_proto_aug_dict(save_path)

            '''save results dict after each session'''
            best_acc_list_dict = {
                'best_test_acc_all_list': args.best_test_acc_all_list,
                'best_test_acc_old_list': args.best_test_acc_old_list,
                'best_test_acc_new_list': args.best_test_acc_new_list,
                'best_test_acc_soft_all_list': args.best_test_acc_soft_all_list,
                'best_test_acc_seen_list': args.best_test_acc_seen_list,
                'best_test_acc_unseen_list': args.best_test_acc_unseen_list,
            }
            save_results_path = os.path.join(args.model_dir, 'best_acc_list' + '_session-' + str(session+1) + f'.pt')
            args.logger.info('Saving results (best acc list) to {}.'.format(save_results_path))
            torch.save(best_acc_list_dict, save_results_path)

        # print final results
        args.logger.info('\n\n==================== print final results over {} continual sessions ===================='.format(args.continual_session_num))
        for session in range(args.continual_session_num):
            args.logger.info(f'Session-{session+1}: All (Hard): {args.best_test_acc_all_list[session]:.4f} Old: {args.best_test_acc_old_list[session]:.4f} New: {args.best_test_acc_new_list[session]:.4f} | All (Soft): {args.best_test_acc_soft_all_list[session]:.4f} Seen: {args.best_test_acc_seen_list[session]:.4f} Unseen: {args.best_test_acc_unseen_list[session]:.4f}')
        for session in range(args.continual_session_num):
            print(f'Session-{session+1}: All (Hard): {args.best_test_acc_all_list[session]:.4f} Old: {args.best_test_acc_old_list[session]:.4f} New: {args.best_test_acc_new_list[session]:.4f} | All (Soft): {args.best_test_acc_soft_all_list[session]:.4f} Seen: {args.best_test_acc_seen_list[session]:.4f} Unseen: {args.best_test_acc_unseen_list[session]:.4f}')

    else:
        raise NotImplementedError
