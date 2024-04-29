from configs.templates import *
from dotenv import load_dotenv

load_dotenv()
ws_path = os.getenv("WORKSPACE_PATH")

def lung_mil():
    conf = MILconfig()
    conf.nr_feats = 512
    conf.target_label = "Type"
    conf.target_dict = {"Lung_squamous_cell_carcinoma": 0, 
                        "Lung_adenocarcinoma": 1}
    return conf

def msi_mil():
    conf = MILconfig()    
    conf.nr_feats = 512
    conf.target_label = "isMSIH"
    conf.target_dict = {"nonMSIH": 0, "MSIH": 1}
    return conf

def texture100k_autoenc_cls():
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


def tcga_crc_autoenc_cls_msi():
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
    conf.name = "tcga_crc_224_autoenc_cls_msi-nonlinear"
    return conf


def tcga_crc_autoenc_cls_braf():
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
    conf.name = "tcga_crc_224_autoenc_cls_braf"
    return conf


def tcga_crc_autoenc_512_cls():
    conf = tcga_crc_autoenc()
    conf.train_mode = TrainMode.manipulate
    conf.manipulate_mode = "tcga_crc_msi_512"
    conf.manipulate_znormalize = True
    conf.latent_infer_path = f"{ws_path}/mopadi/checkpoints/tcga_crc_512x512/latent.pkl"
    conf.batch_size = 32
    conf.lr = 1e-3
    conf.total_samples = 300_000
    conf.pretrain = PretrainConfig(
        "TCGA CRC 512x512",
        f"{ws_path}/mopadi/checkpoints/tcga_crc_512x512/{tcga_crc_autoenc().name}/last.ckpt",
    )
    conf.name = "tcga_crc_512_autoenc_cls"
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

def brain_autoenc_cls():
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
    conf.name = "brain_autoenc_cls-GBM-IDHmut-3"
    return conf


def pancancer_autoenc_cls():
    conf = pancancer_autoenc()
    conf.train_mode = TrainMode.manipulate
    conf.manipulate_mode = "japan"
    conf.manipulate_znormalize = True
    conf.latent_infer_path = f"{ws_path}/mopadi/checkpoints/pancancer/{pancancer_autoenc().name}/latent.pkl"
    conf.batch_size = 32
    conf.lr = 1e-3
    conf.total_samples = 150_000
    conf.save_every_samples = 1_000
    conf.pretrain = PretrainConfig(
        "Brain",
        f"{ws_path}/mopadi/checkpoints/pancancer/last.ckpt",
    )
    conf.name = "pancancer_cls"
    return conf

 
def lung_subtypes_autoenc_cls():
    conf = pancancer_autoenc()
    conf.train_mode = TrainMode.manipulate
    conf.manipulate_mode = "lung-subtypes"
    conf.manipulate_znormalize = True
    conf.latent_infer_path = f"{ws_path}/mopadi/checkpoints/pancancer/{pancancer_autoenc().name}/latent.pkl"
    conf.batch_size = 32
    conf.lr = 1e-3
    conf.total_samples = 150_000
    conf.save_every_samples = 1_000
    conf.pretrain = PretrainConfig(
        "Lung",
        f"{ws_path}/mopadi/checkpoints/pancancer/last.ckpt",
    )
    conf.name = "lung_subtypes"
    return conf
