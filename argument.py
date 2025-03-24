import argparse
import torch

def get_args():
    parser = argparse.ArgumentParser(description='representational-generation')

    parser.add_argument('--no-wandb', default=False, action='store_true')

    parser.add_argument('--train', default=False, action='store_true', help='train the model')
    parser.add_argument('--trainer', default='scratch', type=str, help='choose the trainer')
    
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n-workers', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--train_GPU_batch_size', type=int, default=256)
    
    # For coupmting MPR
    parser.add_argument('--refer-dataset', type=str, default='fairface', choices=['uniform', 'fairface', 'custom','stable_bias_i','statistics'])
    parser.add_argument('--query-dataset', type=str, default='CLIP')
    parser.add_argument('--dataset-path', type=str, default=None, help='it is only used when query-dataset is general')
    parser.add_argument('--p-ver', type=str, default='v1', help='version of prompts used for generating')
    parser.add_argument('--vision-encoder', type=str, default='CLIP', 
                        choices = ['BLIP', 'CLIP', 'PATHS'])
    parser.add_argument('--target-concept', type=str, default='all')
    parser.add_argument('--target-model', type=str, default='SD_14') #required=True, 
    parser.add_argument('--mpr-group', type=str, nargs='+', default=['gender','age','race'])    
    parser.add_argument('--mpr-onehot', default=False, action='store_true', help='onehot group estimation')
    parser.add_argument('--race-reduce', default=False, action='store_true', help='reduce the category of race')
    parser.add_argument('--n-compute-mpr', type=int, default=1)
    parser.add_argument('--bootstrapping', default=False, action='store_true', help='bootstrapping')
    parser.add_argument('--bal-sampling', default=False, action='store_true', help='balanced sampling over groups1')
    parser.add_argument('--n-resampling', type=int, default=1000, help='bootstrapping')
    parser.add_argument('--resampling-size', default=1000, type=int, help='bootstrapping')
    parser.add_argument('--normalize', default=False, action='store_true', help='normalization for x')
    parser.add_argument('--functionclass', type=str, default='linear', help='functionclass for mpr') # choices=['linear','dt','nn','l2'],    
    
    # For retrieving the dataset
    parser.add_argument('--retrieve', default=False, action='store_true', help='retrieve the dataset')
    parser.add_argument('--retriever', type=str, default='mapr', choices=['mapr', 'random','knn', 'random_ratio'])
    parser.add_argument('--max-depth', type=int, default=2, help='max depth for decision tree')
    parser.add_argument('--ratio', type=float, default=1.0, help='ratio for random_ratio retriever')
    parser.add_argument('--pool-size', type=float, default=1.0)
    parser.add_argument('--refer-size', type=float, default=1.0)
    parser.add_argument('--k', type=int, default=20, help='the number of retrieved sample')

    # mapr hyperparameters
    parser.add_argument('--cutting_planes', type=int, default=50, help='the number of cutting planes in LP')
    parser.add_argument('--n_rhos', type=int, default=30, help='the number of constraint problems')

    # Hyperparameters used for each dataset
    parser.add_argument('--binarize-age', default=True, action='store_false', help='it is used for fairface only')                    

    # hyperparameters for each method
    parser.add_argument('--lamb', type=float, default=0.5, help='regularization strength')

    # Info for result file names 
    parser.add_argument('--date', type=str, default='default', help='date when to save')
    parser.add_argument('--save-dir', type=str, default='results/', help='directory to save the results')

    parser.add_argument('--save-labels', default=False, action='store_true')

    args = parser.parse_args()
    return args
