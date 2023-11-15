from torch.utils.tensorboard import SummaryWriter
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
    parser.add_argument('--pi_s', type=float, default=0.5,
                        help='pi in ssnd framework, proportion of ood data in auxiliary dataset')
    parser.add_argument('--in_shift', type=str, default='gaussian_noise',
                         help='corrupted type of images')
    parser.add_argument("--one_class_idx", type=int, default=3,
                         help='ID class indx')
    
    # Optimization options
    parser.add_argument('--epochs', '-e', type=int, default=10,
                        help='Number of epochs to train.')
    parser.add_argument('--batch_size', '-b', type=int,
                        default=128, help='Batch size.')
    parser.add_argument('--learning_rate', '-lr', type=float,
                        default=0.001, help='The initial learning rate.')
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
    parser.add_argument('--eta', type=float, default=1.0,
                         help='woods with margin loss')
    parser.add_argument('--alpha', type=float, default=0.05,
                         help='number of labeled samples')
    
    parser.add_argument('--mode', type=str, default='multiclass',
                         choices=['multiclass', 'oneclass'],help='number of labeled samples')
    parser.add_argument('--run_index', default=0, type=int, help='run index')
    

    # ALM
    parser.add_argument('--beta_penalty', default=0.1, type=int, help='beta penalty')
    parser.add_argument('--tolerance', default=0.1, type=int, help='threshold tolerance')

    args = parser.parse_args()

    return args


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

    grad = calculate_grads(model)

    args.lambda_1 = args.lambda_1 + args.lambda_1_lr * grad
    args.lambda_2 = args.lambda_2 + args.lambda_2_lr * grad


def calculate_grads(model):
    grads = []

    for param in model.parameters():
        grads.append(torch.mean(param.grad))

    return torch.mean(torch.tensor(grads))




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

        # if beta_1 * (e_in - alpha) + lambda_1 >= 0:
        #     loss += (e_in - alpha) * lambda_1 + (beta_1/2) * torch.pow(e_in, 2)
        if args.beta_1 * e_in + args.lambda_1 >= 0:
            loss += e_in * args.lambda_1 + (args.beta_1/2) * torch.pow(e_in, 2)
        else:
            loss += -(((args.lambda_1) ** 2) / (2 * args.beta_1))

        # if beta_2 * (loss_ce - tou) + lambda_2 >= 0:
        #     loss += (loss_ce - tou) * lambda_2 + (beta_2/2) * torch.pow(loss_ce, 2)
        if args.beta_2 * loss_ce + args.lambda_2 >= 0:
            loss += loss_ce * args.lambda_2 + (args.beta_2/2) * torch.pow(loss_ce, 2)
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


if __name__ == "__main__":
    args = parsing()
    
    args.beta_1 = 0
    args.beta_2 = 0
    args.alpha = 0
    args.lambda_1 = 0
    args.lambda_2 = 0
    args.tou = 0


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
    save_path = './run/exp_' + addr + f'_{args.mode}' + f"__{args.run_index}__" + f'_{args.learning_rate}' + f'_{args.lr_update_rate}' + f'_{args.lr_gamma}' + f'_{args.optimizer}' + '/'
    model_save_path = save_path + 'models/'
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path, exist_ok=True)

    writer = SummaryWriter(save_path)
    global_train_iter = 0
    global_eval_iter = 0
    best_acc = 0.0

    for epoch in range(0, args.epochs):
        print('epoch', epoch + 1, '/', args.epochs)

        losses, model, global_train_iter = train(args, in_train_loader, in_shift_train_loader, aux_train_loader, model, cross_entropy_loss, optimizer, writer, global_train_iter, ALM_optim=False)
        # TODO: calculate loss on all train_data again before calling this function
        losses, model, global_train_iter = train(args, in_train_loader, in_shift_train_loader, aux_train_loader, model, cross_entropy_loss, optimizer, writer, global_train_iter, ALM_optim=True)
        ALM_optimizer(args, model, losses)
        # global_eval_iter, eval_loss, eval_acc, eval_auc = test(val_loader, out_val_loader, model, global_eval_iter, cross_entropy_loss)


        writer.add_scalar("Train/avg_loss", torch.mean(torch.tensor(losses['loss'])), epoch)
        writer.add_scalar("Train/avg_e_in", torch.mean(torch.tensor(losses['e_in'])), epoch)
        writer.add_scalar("Train/avg_e_wild", torch.mean(torch.tensor(losses['e_wild'])), epoch)
        writer.add_scalar("Train/avg_loss_ce", torch.mean(torch.tensor(losses['loss_ce'])), epoch)
        # writer.add_scalar("Train/avg_acc", torch.mean(epoch_accuracies), epoch)
        # writer.add_scalar("Evaluation/avg_loss", np.mean(eval_loss), epoch)
        # writer.add_scalar("Evaluation/avg_acc", np.mean(eval_acc), epoch)
        # writer.add_scalar("Evaluation/avg_auc", np.mean(eval_auc), epoch)

        print(f"Train/avg_loss: {np.mean(epoch_loss)} Train/avg_acc: {np.mean(epoch_accuracies)} \
            Evaluation/avg_loss: {np.mean(eval_loss)} Evaluation/avg_acc: {np.mean(eval_acc)}  Evaluation/avg_auc: {np.mean(eval_auc)}")
        
        if (epoch+1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(model_save_path, f'model_params_epoch_{epoch}.pt'))

        if np.mean(eval_acc) > best_acc:
            best_acc = np.mean(eval_acc)
            torch.save(model.state_dict(), os.path.join(model_save_path, 'best_params.pt'))
        
        if np.mean(eval_acc) < best_acc:
            scheduler.step()
        

