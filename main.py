from evaluate import compare_models
from objectives import get_objective_func
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='symmetric')
    
    parser.add_argument('--input_dim', type=int, default=10, help='input dimension of set elements')
    parser.add_argument('--training_set_size', type=int, default=4, help='Number of elements per set during training')
    parser.add_argument('--hidden_dim', type=int, default=100, help='hidden dimension of all models')
    parser.add_argument('--iterations', type=int, default=5000, help='total number of iterations during training')
    parser.add_argument('--batch_size', type=int, default=100, help='batch size during training')
    parser.add_argument('--objective', default="neuron", help='objective function in {mean, median, maximum, softmax, neuron, smooth_neuron}')
    
    args = parser.parse_args()
    
    objective_func = get_objective_func(args.objective, args.input_dim)
    compare_models(args.training_set_size, args.hidden_dim, args.iterations, args.batch_size, args.input_dim , objective_func, narrow = True, log_plot = True)