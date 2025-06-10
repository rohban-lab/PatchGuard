import argparse

from train_model import run_train
from test_model import run_test
from visualize import visualize_heatmap
from download_data import download_dataset
from download_weight import download_weights

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, required=True, default="train", choices=["train", "test", "visualization"])
    
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--class_name", type=str, required=True, default="")
    parser.add_argument("--dataset", type=str, required=True, default="mvtec")
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, default="./")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--test_batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.0008)
    parser.add_argument("--lr_decay_factor", type=float, default=0.0125)
    parser.add_argument("--lr_adaptor", type=float, default=0.0001)
    parser.add_argument("--wd", type=float, default=0.00001)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--no_tqdm", action="store_false", dest='use_tqdm', default=False)

    # feature extractor config
    parser.add_argument("--hf_path", type=str, default='vit_small_patch14_dinov2.lvd142m')
    parser.add_argument("--feature_layers", type=int, nargs='+', default=[12], help="Layers to extract features.")
    parser.add_argument("--reg_layers", type=int, nargs='+', default=[6, 9, 12], help="Layers to apply regularization.")
    # discriminator config
    parser.add_argument("--hidden_dim", type=int, default=2048)
    parser.add_argument("--dsc_layers", type=int, default=1)
    parser.add_argument("--dsc_heads", type=int, default=4)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--smoothing_sigma", type=int, default=6)
    parser.add_argument("--smoothing_radius", type=int, default=7)
    # adversarial attack config
    parser.add_argument("--attack_type", type=str, default="PGD")
    parser.add_argument("--no_adv_train", action="store_false", dest='adv_train', default=True)
    parser.add_argument("--no_adv_test", action="store_false", dest='adv_test', default=True)
    parser.add_argument("--epsilon_train", type=float, default=8)
    parser.add_argument("--epsilon_test", type=float, nargs='+', default=[8])
    parser.add_argument("--epsilon_visualization", type=float, default=8)
    parser.add_argument("--step_train", type=int, default=10)
    parser.add_argument("--step_test", type=int, default=10)
    parser.add_argument("--step_visualization", type=int, default=10)
    # regularizer config
    parser.add_argument("--no_reg", action="store_false", dest='use_reg', default=True)
    parser.add_argument("--reg_type", type=str, default="KL_divergence", choices=["KL_divergence"])

    # prepare data and weight
    parser.add_argument("--use_data_prep", action="store_true", default=False)
    parser.add_argument("--use_weight_prep", action="store_true", default=False)


    args = parser.parse_args()
    return args

def main(args):
    if args.use_data_prep:
        download_dataset(args.dataset)
    if args.use_weight_prep:
        download_weights(args.dataset, args.class_name, args.checkpoint_dir)

    if args.mode == "train":
        run_train(args)
    elif args.mode == "test":
        run_test(args)
    else:
        visualize_heatmap(args)

if __name__ == "__main__":
    args = parse_args()
    args.epsilon_train /= 255
    args.epsilon_visualization /= 255
    args.epsilon_test = [epsilon / 255 for epsilon in args.epsilon_test]
    main(args)
