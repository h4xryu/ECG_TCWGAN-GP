import argparse

class Options:
    def __init__(self):
        pass

    def init(self, parser):
        # Global settings
        parser.add_argument('--batch_size', type=int, default=512,
                            help='Batch size for training and validation.')
        parser.add_argument('--nepoch', type=int, default=50,
                            help='Number of training epochs.')
        parser.add_argument('--lr_initial', type=float, default=1e-6,
                            help='Initial learning rate for the optimizer.')
        parser.add_argument('--decay_epoch', type=int, default=20,
                            help='Epoch at which to start decaying the learning rate.')
        parser.add_argument('--gamma', type=float, default=0.001,
                            help='LR scheduler decay factor (used by MultiStepLR).')

        # Device settings
        parser.add_argument('--device', type=str, default='cuda',
                            help='Device to use for training ("cuda" for GPU, "cpu" for CPU).')

        ''' Student Model settings '''
        parser.add_argument('--classes', type=int, default=4,
                            help='Number of output classes for classification.')
        parser.add_argument('--log_name', type=str, default='model',
                            help='Identifier for logging and checkpointing.')
        parser.add_argument('--pretrained_model', type=str,
                            default='./log/model/models/ckpt_opt.pt',
                            help='Path to the pretrained model weights file.')
        

        # ----------------------------
        # Model / architecture
        # (added fields to match training script usage)
        # ----------------------------
        parser.add_argument('--arch', type=str, default='UNet',
                            help='Model architecture name (for logging).')
        parser.add_argument('--head', type=int, default=4,
                            help='Number of attention heads.')
        parser.add_argument('--d_model', type=int, default=180,
                            help='Transformer embedding dimension.')
        parser.add_argument('--d_ff', type=int, default=2,
                            help='Transformer feed-forward dimension.')
        parser.add_argument('--num_layers', type=int, default=1,
                            help='Number of transformer layers.')
        parser.add_argument('--drop_out', type=float, default=0.2,
                            help='Dropout rate.')

        # ----------------------------
        # Knowledge distillation
        # (names/usage align with training script/log_dir)
        # ----------------------------
        parser.add_argument('--KD_weight', type=float, default=0.1,
                            help='Weight for KD loss/branch.')
        parser.add_argument('--CLT_weight', type=float, default=0.1,
                            help='Weight for CLT loss/branch.')

        # Dataset settings
        parser.add_argument('--fs', type=int, default=360,
                            help='Sampling frequency of the ECG data.')
        parser.add_argument('--path_train_npz', type=str,
                            default='./dataset/mitbih_interpatient/mitbih_train.npz',
                            help='Path to save the training data.')
        parser.add_argument('--path_val_npz', type=str,
                            default='./dataset/mitbih_interpatient/mitbih_test.npz',
                            help='Path to save the training labels.')

        return parser

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Options for ECG Classification Training')
    opt = Options().init(parser).parse_args()
    print(opt)
