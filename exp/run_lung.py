from configs.templates import *
from configs.templates_latent import *
from configs.templates_cls import *
from train_diff_autoenc import *
from linear_clf.train_linear_cls import *

if __name__ == '__main__':

    # train linear latent classifier
    gpus = [0]
    conf = lung_linear_cls()
    train_cls(conf, gpus=gpus)
