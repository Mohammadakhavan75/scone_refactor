from torch.utils.tensorboard import SummaryWriter
from models.wrn import WideResNet
from datetime import datetime
import numpy as np
import argparse
import torch
import os


def parsing():
    parser = argparse.ArgumentParser(description='Tunes a CIFAR Classifier with OE',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # datasets
    parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100'],
                        default='cifar10', help='Choose between CIFAR-10, CIFAR-100, MNIST.')
    parser.add_argument('--aux_out_dataset', type=str, default='svhn', choices=['svhn'],
                        help='Auxiliary out of distribution dataset')
    parser.add_argument('--test_out_dataset', type=str, default='svhn', choices=['svhn'],
                        help='Test out of distribution dataset')
    parser.add_argument('--pi_1', type=float, default=0.5,
                        help='pi in ssnd framework, proportion of ood data in auxiliary dataset')
    parser.add_argument('--pi_2', type=float, default=0.5,
                        help='pi in ssnd framework, proportion of ood data in auxiliary dataset')
    parser.add_argument('--shift', type=str, default='gaussian_noise',
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
    parser.add_argument('--optim', type=str, default='sgd',
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
    
    args = parser.parse_args()

    return args


def load_optim(args, model):
    
    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate,
                                     momentum=args.momentum,weight_decay=args.decay)
    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate, 
                                     weight_decay=args.decay)
    else:
        raise NotImplemented("Not implemented optimizer!")
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_update_rate, gamma=args.lr_gamma)

    return optimizer, scheduler



if __name__ == "__main__":
    args = parsing()

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
    
    optimizer, scheduler = load_optim(args, model)

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
    train_global_iter = 0
    global_eval_iter = 0
    best_acc = 0.0
    
    for epoch in range(0, args.epochs):
        print('epoch', epoch + 1, '/', args.epochs)

        train_global_iter, epoch_loss, epoch_accuracies = train(train_loader, out_train_loader, net, train_global_iter, criterion, optimizer, sigmoid, device)
        global_eval_iter, eval_loss, eval_acc, eval_auc = test(val_loader, out_val_loader, net, global_eval_iter, criterion, device, sigmoid)


        writer.add_scalar("Train/avg_loss", np.mean(epoch_loss), epoch)
        writer.add_scalar("Train/avg_acc", np.mean(epoch_accuracies), epoch)
        writer.add_scalar("Evaluation/avg_loss", np.mean(eval_loss), epoch)
        writer.add_scalar("Evaluation/avg_acc", np.mean(eval_acc), epoch)
        writer.add_scalar("Evaluation/avg_auc", np.mean(eval_auc), epoch)

        print(f"Train/avg_loss: {np.mean(epoch_loss)} Train/avg_acc: {np.mean(epoch_accuracies)} \
            Evaluation/avg_loss: {np.mean(eval_loss)} Evaluation/avg_acc: {np.mean(eval_acc)}  Evaluation/avg_auc: {np.mean(eval_auc)}")
        
        if (epoch+1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(model_save_path, f'model_params_epoch_{epoch}.pt'))

        if np.mean(eval_acc) > best_acc:
            best_acc = np.mean(eval_acc)
            torch.save(model.state_dict(), os.path.join(model_save_path, 'best_params.pt'))
        
        if np.mean(eval_acc) < best_acc:
            scheduler.step()
        