from configs.templates import *
from dotenv import load_dotenv

load_dotenv()
ws_path = os.getenv("WORKSPACE_PATH")

def lung_mil():
    conf = MILconfig()
    conf.nr_feats = 512
    conf.num_epochs = 50
    conf.sab_ln = False
    conf.target_label = "Type"
    conf.target_dict = {"Lung_squamous_cell_carcinoma": 0, 
                        "Lung_adenocarcinoma": 1}
    return conf

def crc_mil():
    conf = MILconfig()    
    conf.nr_feats = 512
    conf.num_epochs = 150
    conf.sab_ln = False
    conf.target_label = "isMSIH"
    conf.target_dict = {"nonMSIH": 0, "MSIH": 1}
    return conf

def brca_mil():
    conf = MILconfig()    
    conf.nr_feats = 512
    conf.num_epochs = 200
    conf.sab_ln = False
    #conf.target_label = "PIK3CA_driver"
    conf.target_label = "BRCA_Pathology"
    #conf.target_label = "TP53"HRD_binary
    #conf.target_label = "HRD_binary"
    #conf.target_dict = {"HRD_negative": 0, "HRD_positive": 1}
    #conf.target_dict = {"WT": 0, "MUT": 1}
    #conf.target_dict = {"NODRIVER": 0, "DRIVER": 1}
    conf.target_dict = {"IDC": 0, "ILC": 1}
    return conf

def liver_types_mil():
    conf = MILconfig()    
    conf.nr_feats = 512
    conf.target_label = "Type"
    conf.num_epochs = 140
    conf.sab_ln = False
    conf.target_dict = {"hcc": 0, "cca": 1}
    return conf

def brain_mil():
    conf = MILconfig()    
    conf.nr_feats = 512
    conf.target_label = "2021WHO"
    conf.num_epochs = 100
    conf.sab_ln = False
    conf.target_dict = {"GBM_WT": 0, "A4_IDH": 1}
    return conf


def texture100k_linear_cls():
    conf = texture100k_autoenc()
    conf.train_mode = TrainMode.manipulate
    conf.manipulate_mode = "texture"
    conf.manipulate_znormalize = True
    conf.latent_infer_path = f"{ws_path}/mopadi/checkpoints/texture100k/latent.pkl"
    conf.batch_size = 32
    conf.lr = 1e-3
    conf.total_samples = 300_000
    conf.pretrain = PretrainConfig(
        "Texture100k",
        f"{ws_path}/mopadi/checkpoints/texture100k/{texture100k_autoenc().name}/last.ckpt",
    )
    conf.name = "texture100k_autoenc_cls"
    return conf


def tcga_crc_linear_cls_msi():
    conf = tcga_crc_autoenc()
    conf.train_mode = TrainMode.manipulate
    conf.manipulate_mode = "tcga_crc_msi"
    conf.manipulate_znormalize = True
    conf.latent_infer_path = f"{ws_path}/mopadi/checkpoints/tcga_crc_224x224/{tcga_crc_autoenc().name}/latent.pkl"
    conf.batch_size = 124
    conf.lr = 1e-4
    conf.total_samples = 2_000_000
    conf.pretrain = PretrainConfig(
        "TCGA CRC 224x224",
        f"{ws_path}/mopadi/checkpoints/tcga_crc_224x224/{tcga_crc_autoenc().name}/last.ckpt",
    )
    conf.name = "tcga_crc_224_cls_msi-nonlinear"
    return conf


def tcga_crc_linear_cls_braf():
    conf = tcga_crc_autoenc()
    conf.train_mode = TrainMode.manipulate
    conf.manipulate_mode = "tcga_crc_braf"
    conf.manipulate_znormalize = True
    conf.latent_infer_path = f"{ws_path}/mopadi/checkpoints/tcga_crc_224x224/{tcga_crc_autoenc().name}/latent.pkl"
    conf.batch_size = 32
    conf.lr = 1e-3
    conf.total_samples = 300_000
    conf.pretrain = PretrainConfig(
        "TCGA CRC 224x224",
        f"{ws_path}/mopadi/checkpoints/tcga_crc_224x224/{tcga_crc_autoenc().name}/last.ckpt",
    )
    conf.name = "tcga_crc_224_cls_braf"
    return conf


def tcga_crc_linear_512_cls():
    conf = tcga_crc_autoenc()
    conf.train_mode = TrainMode.manipulate
    conf.manipulate_mode = "tcga_crc_msi"
    conf.manipulate_znormalize = True
    conf.latent_infer_path = f"{ws_path}/mopadi/checkpoints/crc/tcga_crc_512_autoenc/latent.pkl"
    conf.batch_size = 32
    conf.lr = 1e-3
    conf.total_samples = 300_000
    conf.pretrain = PretrainConfig(
        "TCGA CRC 512x512",
        f"{ws_path}/mopadi/checkpoints/crc/tcga_crc_512_autoenc/last.ckpt",
    )
    conf.name = "tcga_crc_512_linear_cls"
    return conf

def tcga_brca_autoenc_512_cls():
    conf = tcga_brca_autoenc()
    conf.train_mode = TrainMode.manipulate
    conf.manipulate_mode = ManipulateMode.tcga_brca_all
    conf.manipulate_znormalize = True
    conf.latent_infer_path = f"{ws_path}/mopadi/{tcga_brca_autoenc().name}/latent.pkl"
    conf.batch_size = 32 
    conf.lr = 1e-3
    conf.total_samples = 1_500_000
    # use the pretraining trick instead of continuing trick
    conf.pretrain = PretrainConfig(
        '200M',
        f'{ws_path}/mopadi/{tcga_brca_autoenc().name}/last.ckpt',
    )
    conf.name = 'tcga_brca_512_autoenc_cls'
    return conf

def brain_linear_cls():
    conf = brain_autoenc()
    conf.train_mode = TrainMode.manipulate
    conf.manipulate_mode = "brain"
    conf.manipulate_znormalize = True
    conf.latent_infer_path = f"{ws_path}/mopadi/checkpoints/brain/{brain_autoenc().name}/latent.pkl"
    conf.batch_size = 32
    conf.lr = 1e-3
    conf.total_samples = 150_000
    conf.save_every_samples = 1_000
    conf.pretrain = PretrainConfig(
        "Brain",
        f"{ws_path}/mopadi/checkpoints/brain/last.ckpt",
    )
    conf.name = "brain_linear_cls-GBM-IDHmut-3"
    return conf


def pancancer_linear_cls():
    conf = pancancer_autoenc()
    conf.train_mode = TrainMode.manipulate
    conf.manipulate_mode = "pancancer"
    conf.manipulate_znormalize = True
    conf.latent_infer_path = f"{ws_path}/mopadi/checkpoints/pancancer/autoenc/latent.pkl"
    conf.batch_size = 32
    conf.lr = 1e-3
    conf.total_samples = 150_000
    conf.save_every_samples = 1_000
    conf.pretrain = PretrainConfig(
        "Pancancer",
        f"{ws_path}/mopadi/checkpoints/pancancer/autoenc/last.ckpt",
    )
    conf.name = "pancancer_linear_cls"
    return conf

 
def lung_linear_cls():
    conf = pancancer_autoenc()
    conf.train_mode = TrainMode.manipulate
    conf.manipulate_mode = "lung"
    conf.manipulate_znormalize = True
    conf.latent_infer_path = f"{ws_path}/mopadi/checkpoints/pancancer/autoenc/latent.pkl"
    conf.batch_size = 32
    conf.lr = 1e-3
    conf.total_samples = 150_000
    conf.save_every_samples = 1_000
    conf.pretrain = PretrainConfig(
        "Pancancer",
        f"{ws_path}/mopadi/checkpoints/pancancer/autoenc/last.ckpt",
    )
    conf.name = "lung_subtypes_linear__cls"
    return conf


def liver_cancer_types_cls():
    conf = pancancer_autoenc()
    conf.train_mode = TrainMode.manipulate
    conf.manipulate_mode = "liver_cancer_types"
    conf.manipulate_znormalize = True
    conf.latent_infer_path = f"{ws_path}/mopadi/checkpoints/pancancer/autoenc/latent.pkl"
    conf.batch_size = 32
    conf.lr = 1e-3
    conf.total_samples = 150_000
    conf.save_every_samples = 1_000
    conf.pretrain = PretrainConfig(
        "Pancancer",
        f"{ws_path}/mopadi/checkpoints/pancancer/autoenc/last.ckpt",
    )
    conf.name = "liver_cancer_types_linear_cls"
    return conf
