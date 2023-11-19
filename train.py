from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import det_curve, accuracy_score, roc_auc_score
from models.wrn import WideResNet
from datetime import datetime
from tqdm import tqdm
import numpy as np
import argparse
import torch
import os

from dataset.datasets import main



def parsing():
    parser = argparse.ArgumentParser(description='Tunes a CIFAR Classifier with OE',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # datasets
    parser.add_argument('--in_dataset', type=str, choices=['cifar10', 'cifar100'],
                        default='cifar10', help='Choose between CIFAR-10, CIFAR-100, MNIST.')
    parser.add_argument('--aux_dataset', type=str, default='svhn', choices=['svhn'],
                        help='Auxiliary out of distribution dataset')
    parser.add_argument('--ood_dataset', type=str, default='svhn', choices=['svhn'],
                        help='Test out of distribution dataset')
    parser.add_argument('--pi_c', type=float, default=0.5,
                        help='pi in ssnd framework, proportion of ood data in auxiliary dataset')
    parser.add_argument('--pi_s', type=float, default=0.1,
                        help='pi in ssnd framework, proportion of ood data in auxiliary dataset')
    parser.add_argument('--in_shift', type=str, default='gaussian_noise',
                         help='corrupted type of images')
    parser.add_argument("--one_class_idx", type=int, default=3,
                         help='ID class indx')
    
    # Optimization options
    parser.add_argument('--epochs', '-e', type=int, default=50,
                        help='Number of epochs to train.')
    parser.add_argument('--batch_size', '-b', type=int,
                        default=128, help='Batch size.')
    parser.add_argument('--learning_rate', '-lr', type=float,
                        default=0.1, help='The initial learning rate.')
    parser.add_argument('--lr_update_rate', type=float, default=5,
                         help='The update rate for learning rate.')
    parser.add_argument('--lr_gamma', type=float, default=0.9,
                         help='The gamma param for updating learning rate.')
    parser.add_argument('--optimizer', type=str, default='sgd',
                         help='The initial learning rate.')
    parser.add_argument('--momentum', type=float, default=0.9,
                         help='Momentum.')
    parser.add_argument('--decay', '-d', type=float,
                        default=0.0005, help='Weight decay (L2 penalty).')
    
    # WRN Architecture
    parser.add_argument('--layers', default=40, type=int,
                        help='total number of layers')
    parser.add_argument('--widen-factor', default=2, type=int,
                         help='widen factor')
    parser.add_argument('--droprate', default=0.3, type=float,
                        help='dropout probability')
    
    # Checkpoints
    parser.add_argument('--load_pretrained', type=str, default=None,
                         help='Load pretrained model to test or resume training.')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='start epoches')

    # Acceleration
    parser.add_argument('--device', type=str, default='cuda',
                         help='cuda or cpu.')
    parser.add_argument('--gpu_id', type=int, default=0,
                         help='Which GPU to run on.')
    parser.add_argument('--num_worker', type=int, default=4,
                        help='Pre-fetching threads.')
    
    # Loss function parameters
    parser.add_argument('--eta', type=float, default=1,
                         help='margin loss')
    parser.add_argument('--alpha', type=float, default=0.05,
                         help='number of labeled samples')
    parser.add_argument('--tou', type=float, default=2,
                         help='number of labeled samples')
    parser.add_argument('--T', default=1., type=float,
                         help='temperature: energy|Odin')  # T = 1 suggested by energy paper
    parser.add_argument('--mode', type=str, default='multiclass',
                         choices=['multiclass', 'oneclass'],help='number of labeled samples')
    parser.add_argument('--run_index', default=0, type=int, help='run index')
    
    # ALM
    parser.add_argument('--lambda_1', default=0, type=int,
                         help='Initial lambda_1 value')
    parser.add_argument('--lambda_2', default=0, type=int,
                         help='Initial lambda_1 value')
    parser.add_argument('--lambda_1_lr', default=1, type=int,
                         help='lambda_1 learning rate')
    parser.add_argument('--lambda_2_lr', default=1, type=int,
                         help='lambda_1 learning rate')
    parser.add_argument('--beta_1', default=1, type=int,
                         help='beta_1 parameter')
    parser.add_argument('--beta_2', default=1, type=int,
                         help='beta_2 beta_1')
    parser.add_argument('--beta_penalty', default=1.5, type=int,
                         help='beta penalty')
    parser.add_argument('--tolerance', default=0, type=int,
                         help='threshold tolerance')

    args = parser.parse_args()

    return args


def tensor_to_np(x):
    return x.data.cpu().numpy()


def load_optim(args, model):
    
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate,
                                     momentum=args.momentum,weight_decay=args.decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate, 
                                     weight_decay=args.decay)
    else:
        raise NotImplemented("Not implemented optimizer!")
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_update_rate, gamma=args.lr_gamma)

    return optimizer, scheduler


def make_wild_data(args, in_train_batch, in_shift_train_batch, aux_train_batch):
    
    max_length = np.min([len(in_train_batch[0]), len(in_shift_train_batch[0]), len(aux_train_batch[0])])
    b_size = args.batch_size
    if max_length < args.batch_size:
        b_size = max_length
    
    rng = np.random.default_rng()
    
    mask_12 = rng.choice(a=[False, True], size=(b_size,), p=[1 - (args.pi_c + args.pi_s), (args.pi_c + args.pi_s)])
    in_train_batch_subsampled = in_train_batch[0][np.invert(mask_12)]
    in_train_batch_subsampled_label = in_train_batch[1][np.invert(mask_12)]

    mask_1 = rng.choice(a=[False, True], size=(b_size,), p=[1 - args.pi_c, args.pi_c])
    in_shift_train_batch_subsampled = in_shift_train_batch[0][mask_1]
    in_shift_train_batch_subsampled_label = in_shift_train_batch[1][mask_1]

    mask_2 = rng.choice(a=[False, True], size=(b_size,), p=[1 - args.pi_s, args.pi_s])
    aux_train_batch_subsampled = aux_train_batch[0][mask_2]
    aux_train_batch_subsampled_label = aux_train_batch[1][mask_2]

    wild_data_imgs = torch.cat((in_train_batch_subsampled, in_shift_train_batch_subsampled, aux_train_batch_subsampled), 0)
    wild_data_lables = torch.cat((in_train_batch_subsampled_label, in_shift_train_batch_subsampled_label, aux_train_batch_subsampled_label), 0)

    return wild_data_imgs, wild_data_lables


def energy_T(data):
    return list(-tensor_to_np((args.T * torch.logsumexp(data / args.T, dim=1))))


def energy_test(data, learnable_parameter_w):
    return torch.mean(torch.sigmoid(learnable_parameter_w(
                (torch.logsumexp(data, dim=1)).unsqueeze(1)).squeeze()))


def energy_wild(out, learnable_parameter_w, len_wild):
    return torch.mean(torch.sigmoid(learnable_parameter_w(
                (torch.logsumexp(out[-len_wild:], dim=1)).unsqueeze(1)).squeeze()))


def energy_in(out, learnable_parameter_w, len_in, args):
    return torch.mean(torch.sigmoid(-learnable_parameter_w(
                (torch.logsumexp(out[:len_in], dim=1) - args.eta).unsqueeze(1)).squeeze()))
    

def ALM_optimizer(args, model, losses):
    
    if torch.mean(torch.tensor(losses['e_in'])) > args.alpha + args.tolerance:
        args.beta_1 *= args.beta_penalty
    if torch.mean(torch.tensor(losses['loss_ce'])) > args.tou + args.tolerance:
        args.beta_2 *= args.beta_penalty

    args.lambda_1 = args.lambda_1 + args.lambda_1_lr * grads_lambda_1(args, losses)
    args.lambda_2 = args.lambda_2 + args.lambda_2_lr * grads_lambda_2(args, losses)


def grads_lambda_1(args, losses):
    if args.beta_1 * torch.mean(torch.tensor(losses['e_in'])) + args.lambda_1 >= 0:
        return torch.mean(torch.tensor(losses['e_in']))
    else: 
        return -2 * args.lambda_1/(2*args.beta_1)


def grads_lambda_2(args, losses):
    if args.beta_2 * torch.mean(torch.tensor(losses['loss_ce'])) + args.lambda_2 >= 0:
        return torch.mean(torch.tensor(losses['loss_ce']))
    else:
        return -2 * args.lambda_2/(2*args.beta_2)


def processing_auroc(out_scores, in_scores):
    in_labels = np.zeros(len(in_scores))
    out_labels = np.ones(len(out_scores))
    y_true = np.concatenate([in_labels, out_labels])
    y_score = np.concatenate([in_scores, out_scores])
    auroc = roc_auc_score(y_true=y_true, y_score=y_score)

    return auroc


def compute_fnr(out_scores, in_scores, fpr_cutoff=.05):
    '''
    compute fnr at 05
    '''

    in_labels = np.zeros(len(in_scores))
    out_labels = np.ones(len(out_scores))
    y_true = np.concatenate([in_labels, out_labels])
    y_score = np.concatenate([in_scores, out_scores])
    fpr, fnr, thresholds = det_curve(y_true=y_true, y_score=y_score)

    idx = np.argmin(np.abs(fpr - fpr_cutoff))

    fpr_at_fpr_cutoff = fpr[idx]
    fnr_at_fpr_cutoff = fnr[idx]

    if fpr_at_fpr_cutoff > 0.1:
        fnr_at_fpr_cutoff = 1.0

    return fnr_at_fpr_cutoff


def train(args, in_train_loader, in_shift_train_loader, aux_train_loader, model, cross_entropy_loss, optimizer, writer, global_train_iter, ALM_optim=False):
    
    optimizer.zero_grad()
    losses = {
        'loss': [],
        'e_wild': [],
        'e_in': [],
        'loss_ce': []
        }
    
    loader = zip(in_train_loader, in_shift_train_loader, aux_train_loader)
    for data in tqdm(loader):
        in_train_batch, in_shift_train_batch, aux_train_batch = data
        wild_data_imgs, wild_data_lables  = make_wild_data(args, in_train_batch, in_shift_train_batch, aux_train_batch)
        in_data_imgs, in_data_lables = in_train_batch
        in_data_imgs, in_data_lables, wild_data_imgs, wild_data_lables = \
            in_data_imgs.to(args.device), in_data_lables.to(args.device), wild_data_imgs.to(args.device), wild_data_lables.to(args.device)

        data = torch.cat((in_data_imgs, wild_data_imgs), 0)

        if not ALM_optim:
            optimizer.zero_grad()
            global_train_iter += 1

        out = model(data)
        e_wild = energy_wild(out, args.learnable_parameter_w, len(wild_data_imgs))
        e_in = energy_in(out, args.learnable_parameter_w, len(in_data_imgs), args)

        loss_ce = cross_entropy_loss(out[:len(in_data_imgs)], in_data_lables)

        # Calcualting loss function using ALM
        loss = e_wild

        if args.beta_1 * (e_in - args.alpha) + args.lambda_1 >= 0:
            loss += (e_in - args.alpha) * args.lambda_1 + (args.beta_1/2) * torch.pow(e_in, 2)
        else:
            loss += -(((args.lambda_1) ** 2) / (2 * args.beta_1))

        if args.beta_2 * (loss_ce - args.tou) + args.lambda_2 >= 0:
            loss += (loss_ce - args.tou) * args.lambda_2 + (args.beta_2/2) * torch.pow(loss_ce, 2)
        else:
            loss += -(((args.lambda_2) ** 2) / (2 * args.beta_2))
        
        if not ALM_optim:
            loss.backward()
            optimizer.step()
        else:
            loss.backward()
        
        losses['loss'].append(loss)
        losses['e_wild'].append(e_wild)
        losses['e_in'].append(e_in)
        losses['loss_ce'].append(loss_ce)
        
        if not ALM_optim:
            writer.add_scalar("Train/loss", loss, global_train_iter)
            writer.add_scalar("Train/e_wild", e_wild, global_train_iter)
            writer.add_scalar("Train/e_in", e_in, global_train_iter)
            writer.add_scalar("Train/loss_ce", loss_ce, global_train_iter)


    return losses, model, global_train_iter


def test(args, loader, model, cross_entropy_loss, writer, flag_iter, d_type, scores, e_test, losses):
   
    with torch.no_grad():
        for i, data in enumerate(tqdm(loader)):
            flag_iter += 1
            imgs, lables = data  
            imgs, lables = imgs.to(args.device), lables.to(args.device)

            out = model(imgs)

            if d_type == 'in':
                scores['in'].extend(energy_T(out))
                e_test['e_in_test'].append(energy_test(out, args.learnable_parameter_w))
                loss_ce_in = cross_entropy_loss(out, lables)
                acc['acc_in'].append(accuracy_score(list(tensor_to_np(out.data.max(1)[1])), list(tensor_to_np(lables))))
                test_losses['loss_ce'].append(loss_ce_in)
                writer.add_scalar("Evaluation/loss_ce_in", loss_ce_in, flag_iter)
                writer.add_scalar("Evaluation/e_in_test", e_test['e_in_test'][i], flag_iter)
                writer.add_scalar("Evaluation/score_in", scores['in'][i], flag_iter)

            if d_type == 'shift':
                scores['shift'].extend(energy_T(out))
                e_test['e_shift_test'].append(energy_test(out, args.learnable_parameter_w))
                loss_ce_shift = cross_entropy_loss(out, lables)
                acc['acc_shift'].append(accuracy_score(list(tensor_to_np(out.data.max(1)[1])), list(tensor_to_np(lables))))
                test_losses['loss_ce_shift'].append(loss_ce_shift)
                writer.add_scalar("Evaluation/loss_ce_shift", loss_ce_shift, flag_iter)
                writer.add_scalar("Evaluation/e_shift_test", e_test['e_shift_test'][i], flag_iter)
                writer.add_scalar("Evaluation/score_shift", scores['shift'][i], flag_iter)

            if d_type == 'aux':
                scores['aux'].extend(energy_T(out))
                e_test['e_aux_test'].append(energy_test(out, args.learnable_parameter_w))
                writer.add_scalar("Evaluation/e_aux_test", e_test['e_aux_test'][i], flag_iter)
                writer.add_scalar("Evaluation/score_aux", scores['aux'][i], flag_iter)

            if d_type == 'ood':
                scores['ood'].extend(energy_T(out))
                e_test['e_ood_test'].append(energy_test(out, args.learnable_parameter_w))
                writer.add_scalar("Evaluation/e_ood_test", e_test['e_ood_test'][i], flag_iter)
                writer.add_scalar("Evaluation/score_ood", scores['ood'][i], flag_iter)

    return flag_iter



if __name__ == "__main__":
    args = parsing()

    in_train_loader, in_test_loader, in_shift_train_loader, in_shift_test_loader, aux_train_loader, aux_test_loader, ood_test_loader = main(args)
    
    if args.mode == 'multiclass':
        num_classes = 10
    elif args.mode == 'oneclass':
        num_classes = 1
    else:
        raise NotImplemented("mode must be multiclass or oneclass !!")

    model = WideResNet(args.layers, num_classes, args.widen_factor, dropRate=args.droprate)
    learnable_parameter_w = torch.nn.Linear(1, 1)

    model = model.to(args.device)
    learnable_parameter_w = learnable_parameter_w.to(args.device)
    
    args.learnable_parameter_w = learnable_parameter_w

    optimizer, scheduler = load_optim(args, model)
    cross_entropy_loss = torch.nn.CrossEntropyLoss()

    if args.load_pretrained is not None:
        print("Loading from pretrained model...")
        model.load_state_dict(torch.load(args.load_pretrained))
        
    #     save_path = args.save_path
    #     model_save_path = save_path + 'models/'
    # else:
    addr = datetime.today().strftime('%Y_%m_%d_%H_%M_%S_%f')
    save_path = './run/exp_' + addr + '/'
    model_save_path = save_path + 'models/'
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path, exist_ok=True)

    with open(save_path + 'config.txt', 'w') as f:
        f.write(f'mode: {args.mode}\n\
                run index: {args.run_index}\n\
                learning_rate: {args.learning_rate}\n\
                lr_update_rate: {args.lr_update_rate}\n\
                lr_gamma: {args.lr_gamma}\n\
                optimizer: {args.optimizer}\
                lambda_1: {args.lambda_1}\n\
                lambda_2: {args.lambda_2}\n\
                lambda_1_lr: {args.lambda_1_lr}\n\
                lambda_2_lr: {args.lambda_2_lr}\n\
                beta_1: {args.beta_1}\n\
                beta_2: {args.beta_2}\n\
                beta_penalty: {args.beta_penalty}\n\
                tolerance: {args.tolerance}\n\
                eta: {args.eta}\n\
                tou: {args.tou}\n\
                alpha: {args.alpha}\n\
                T: {args.T}\n\
                pi_c: {args.pi_c}\n\
                pi_s: {args.pi_s}\n\
                in_dataset: {args.in_dataset}\n\
                in_shift: {args.in_shift}\n\
                aux_dataset: {args.aux_dataset}\n\
                ood_dataset: {args.ood_dataset}')
    
    writer = SummaryWriter(save_path)
    global_train_iter = 0
    best_acc = 0.0

    global_test_iter = {
        'in': 0,
        'shift': 0,
        'aux': 0,
        'ood': 0
    }

    lodaers = {
        'in': in_test_loader,
        'shift': in_shift_test_loader,
        'aux': aux_test_loader,
        'ood': ood_test_loader
        }

    for epoch in range(0, args.epochs):
        scores = {
        'in': [],
        'shift': [],
        'aux': [],
        'ood': []
        }

        test_losses = {
            'loss_ce': [],
            'loss_ce_shift': []
            }
        
        acc = {
            'acc_in': [],
            'acc_shift': []
        }

        e_test = {
            'e_in_test': [],
            'e_shift_test': [],
            'e_aux_test': [],
            'e_ood_test': []
        }
        
        print('epoch', epoch + 1, '/', args.epochs)

        losses, model, global_train_iter = train(args, in_train_loader, in_shift_train_loader, aux_train_loader, model, cross_entropy_loss, optimizer, writer, global_train_iter, ALM_optim=False)
        # TODO: What is the difference between using all data or last batch to calculate grads!
        losses, model, global_train_iter = train(args, in_train_loader, in_shift_train_loader, aux_train_loader, model, cross_entropy_loss, optimizer, writer, global_train_iter, ALM_optim=True)
        ALM_optimizer(args, model, losses)
        for d_type in lodaers.keys():
            global_test_iter[d_type] = test(args, lodaers[d_type], model, cross_entropy_loss, writer, global_test_iter[d_type], d_type, scores, e_test, test_losses)

        auroc = processing_auroc(scores['in'], scores['ood'])
        fpr95 = compute_fnr(np.array(scores['in']), np.array(scores['ood']))

        writer.add_scalar("Train_avg/loss", torch.mean(torch.tensor(losses['loss'])), epoch)
        writer.add_scalar("Train_avg/e_in", torch.mean(torch.tensor(losses['e_in'])), epoch)
        writer.add_scalar("Train_avg/e_wild", torch.mean(torch.tensor(losses['e_wild'])), epoch)
        writer.add_scalar("Train_avg/loss_ce", torch.mean(torch.tensor(losses['loss_ce'])), epoch)
        writer.add_scalar("Train_avg/lambda_1", args.lambda_1, epoch)
        writer.add_scalar("Train_avg/lambda_2", args.lambda_2, epoch)
        writer.add_scalar("Train_avg/beta_1", args.beta_1, epoch)
        writer.add_scalar("Train_avg/beta_2", args.beta_2, epoch)
        writer.add_scalar("Evaluation_avg/auroc", auroc, epoch)
        writer.add_scalar("Evaluation_avg/fpr95", fpr95, epoch)

        if (epoch+1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(model_save_path, f'model_params_epoch_{epoch}.pt'))

        if np.mean(acc['acc_in']) > best_acc:
            best_acc = np.mean(acc['acc_in'])
            torch.save(model.state_dict(), os.path.join(model_save_path, 'best_params.pt'))
        
        if np.mean(acc['acc_in']) < best_acc:
            scheduler.step()
        

