import argparse

def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', dest='project_name', type=str, default='review-summarizer',
                        help='Name of project')
    parser.add_argument('--random_seed', dest='random_seed', type=int, default=42,
                        help='Random Seed')
    parser.add_argument('--bart_type', dest='bart_type', type=str, default='facebook/bart-large-cnn',
                        help='The pre-trained bart model to be used')
    parser.add_argument('--load_model_path', dest='load_model_path', type=str, default=None,
                        help='Model Path if not training from scratch')
    parser.add_argument('--save_model_path', dest='save_model_path', type=str, default=None,
                        help='Path for saving model')
    parser.add_argument('--data_path', dest='data_path', type=str, default=None,
                        help='Dataset Path')
    parser.add_argument('--data_size', dest='data_size', type=int, default=50000,
                        help='Dataset Path')                   
    parser.add_argument('--local_run', dest='local_run', action='store_true',
                        default=False, help='Testing on local computer?')
    parser.add_argument('--from_scratch', dest='from_scratch', action='store_true',
                        default=False, help='Train from scratch or not?')
    parser.add_argument('--max_len', dest='max_len', type=int, default=512,
                        help='Maximum input length for encoder')
    parser.add_argument('--max_output_length', dest='max_output_length', type=int, default=512,
                        help='Maximum output length for decoder')
    parser.add_argument('--train_batch_size', dest='train_batch_size', type=int, default=4,
                        help='Training batch size')
    parser.add_argument('--predict_batch_size', dest='predict_batch_size', type=int, default=32,
                        help='Prediction batch size')
    parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=1e-5,
                        help='Learning rate')
    parser.add_argument('--warmup_proportion', dest='warmup_proportion', type=float, default=0.01,
                        help='Warmup proportion')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=0.0,
                        help='Weight decay')
    parser.add_argument('--adam_epsilon', dest='adam_epsilon', type=float, default=1e-8,
                        help='Adam Epsilon')
    parser.add_argument('--max_grad_norm', dest='max_grad_norm', type=float, default=1.0,
                        help='Max grad norm')
    parser.add_argument('--gradient_accumulation_steps', dest='gradient_accumulation_steps', type=int, default=1,
                        help='gradient accumulation steps')
    parser.add_argument('--num_train_epochs', dest='num_train_epochs', type=int, default=1,
                        help='Number of training epochs')
    parser.add_argument('--warmup_steps', dest='warmup_steps', type=int, default=0,
                        help='Warmup steps')
    parser.add_argument('--wait_step', dest='wait_step', type=int, default=10,
                        help='Wait step')
    parser.add_argument('--num_beams', dest='num_beams', type=int, default=4,
                        help='Number of beams')
    parser.add_argument('--exp_type', dest='exp_type', type=str, default=None,
                        help='Category of experiment to run: "train_bart", "test_bart"')
    # Model Evaluation
    parser.add_argument('--output_json_path', dest='output_json_path', type=str, default='output_bart.json',
                        help='Directory for saving output json file')
    args = parser.parse_args()
    return args