import argparse

def get_params():
    parser = argparse.ArgumentParser()
    # LSTM and CNN Config
    parser.add_argument('--max_voc_size', dest='max_voc_size', type=int, default=25_000,
                        help='Maximum vocabulary size')
    parser.add_argument('--emb_dim', dest='emb_dim', type=int, default=100,
                        help='Embedding size')
    parser.add_argument('--dropout', dest='dropout', type=int, default=0.0,
                        help='Dropout')
    parser.add_argument('--lstm_hid_dim', dest='lstm_hid_dim', type=int, default=256,
                        help='LSTM hidden dimension')
    parser.add_argument('--n_lstm_layers', dest='n_lstm_layers', type=int, default=10,
                        help='Number of LSTM layers')
    parser.add_argument('--with_attention', dest='with_attention', action='store_true',
                        default=False, help='Attention mechanism included or not?')
    parser.add_argument('--n_filters', dest='n_filters', type=int, default=100,
                        help='Number of filters')
    parser.add_argument('--filter_sizes', dest='filter_sizes', type=str, default='345',
                        help='size of cnn filters')
    parser.add_argument('--load_model_path', dest='load_model_path', type=str, default=None,
                        help='path for loading model')
    parser.add_argument('--load_emb_path', dest='load_emb_path', type=str, default=None,
                        help='path for loading embeddings')
    parser.add_argument('--load_voc_path', dest='load_voc_path', type=str, default=None,
                        help='path for loading vocabulary')
    parser.add_argument('--save_model_path', dest='save_model_path', type=str, default=None,
                        help='path for saving model')
    parser.add_argument('--save_emb_path', dest='save_emb_path', type=str, default=None,
                        help='path for saving embeddings')
    parser.add_argument('--save_voc_path', dest='save_voc_path', type=str, default=None,
                        help='path for saving vocabulary')
    parser.add_argument('--data_dir', dest='data_dir', type=str, default=None,
                        help='data parent directory')
    parser.add_argument('--train_file_name', dest='train_file_name', type=str, default=None,
                        help='name of train data file')
    parser.add_argument('--val_file_name', dest='val_file_name', type=str, default=None,
                        help='name of validation data file')
    parser.add_argument('--test_file_name', dest='test_file_name', type=str, default=None,
                        help='name of test data file')
    # BART Config
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
    parser.add_argument('--warmup_steps', dest='warmup_steps', type=int, default=0,
                        help='Warmup steps')
    parser.add_argument('--wait_step', dest='wait_step', type=int, default=10,
                        help='Wait step')
    parser.add_argument('--num_beams', dest='num_beams', type=int, default=4,
                        help='Number of beams')
    parser.add_argument('--max_len', dest='max_len', type=int, default=512,
                        help='Maximum input length for encoder')
    parser.add_argument('--max_output_length', dest='max_output_length', type=int, default=512,
                        help='Maximum output length for decoder')
    parser.add_argument('--bart_type', dest='bart_type', type=str, default='facebook/bart-large',
                        help='The pre-trained bart model to be used')
    parser.add_argument('--load_bart_path', dest='load_bart_path', type=str, default=None,
                        help='BART Path if not training from scratch')
    parser.add_argument('--load_classifier_path', dest='load_classifier_path', type=str, default=None,
                        help='Path for loading classifier head')
    parser.add_argument('--save_bart_path', dest='save_bart_path', type=str, default=None,
                        help='Path for saving model')
    parser.add_argument('--save_classifier_path', dest='save_classifier_path', type=str, default=None,
                        help='Path for saving classifier head')
    parser.add_argument('--data_path', dest='data_path', type=str, default=None,
                        help='Dataset Path')
    parser.add_argument('--data_size', dest='data_size', type=int, default=50000,
                        help='Dataset Size')   
    # General Config
    parser.add_argument('--random_seed', dest='random_seed', type=int, default=1234,
                        help='Random Seed')
    parser.add_argument('--exp_type', dest='exp_type', type=str, default=None,
                        help='Category of experiment to run: "train_lstm", "train_cnn", "train_bart", "test_bart"')
    parser.add_argument('--model_type', dest='model_type', type=str, default=None,
                        help='The model to be used: "lstm", "cnn", "bart"?')
    parser.add_argument('--num_train_epochs', dest='num_train_epochs', type=int, default=1,
                        help='Number of training epochs')
    parser.add_argument('--train_batch_size', dest='train_batch_size', type=int, default=4,
                        help='Training batch size')
    parser.add_argument('--predict_batch_size', dest='predict_batch_size', type=int, default=32,
                        help='Prediction batch size')
    parser.add_argument('--local_run', dest='local_run', action='store_true',
                        default=False, help='Platform used: local or colab')
    parser.add_argument('--local_test', dest='local_test', action='store_true',
                        default=False, help='Testing on local computer?')
    parser.add_argument('--from_scratch', dest='from_scratch', action='store_true',
                        default=False, help='Train from scratch or not?')
    parser.add_argument('--project_name', dest='project_name', type=str, default='answerability verification',
                        help='Name of Project')
    args = parser.parse_args()
    return args