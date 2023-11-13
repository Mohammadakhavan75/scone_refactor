import argparse
from models.wrn import WideResNet

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
    parser.add_argument('--results_dir', type=str,
                        default='results', help='Folder to save .pkl results.')
    parser.add_argument('--checkpoints_dir', type=str,
                        default='checkpoints', help='Folder to save .pt checkpoints.')
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


if __name__ == "__main__":
    args = parsing()
    if args.mode == 'multiclass':
        num_classes = 10
    elif args.mode == 'oneclass':
        num_classes = 1

    model = WideResNet(args.layers, num_classes, args.widen_factor, dropRate=args.droprate)
    print(model)
    