# Code snippets sourced from the Official implementation of Diffusion Autoencoders by Konpat Preechakul
# with modifications by Laura Zigutyte and Tim Lenz
# Original Source: https://github.com/phizaz/diffae
# License: MIT

from configs.config import *
from dataset import *
import os
import copy

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import *
import torch
from dotenv import load_dotenv
from torch.utils.data.dataloader import default_collate
from torch.utils.data import Sampler
import random


load_dotenv()
ws_path = os.getenv("WORKSPACE_PATH")


class ZipLoader:
    def __init__(self, loaders):
        self.loaders = loaders

    def __len__(self):
        return len(self.loaders[0])

    def __iter__(self):
        for each in zip(*self.loaders):
            yield each


class CustomSampler(Sampler):
    def __init__(self, data_source, valid_indices):
        """
        :param data_source: The dataset object
        :param valid_indices: List of indices that should be sampled from the lmdb dataset
        """
        self.data_source = data_source
        self.valid_indices = valid_indices

    def __iter__(self):
        return iter(self.valid_indices)

    def __len__(self):
        return len(self.valid_indices)


class ClsModel(pl.LightningModule):
    def __init__(self, conf: TrainConfig):
        super().__init__()
        assert conf.train_mode.is_manipulate()
        if conf.seed is not None:
            pl.seed_everything(conf.seed)

        self.save_hyperparameters(conf.as_dict_jsonable())
        self.conf = conf
        
        # preparations
        if conf.train_mode == TrainMode.manipulate:
            # this is only important for training!
            # the latent is freshly inferred to make sure it matches the image
            # manipulating latents require the base model
            self.model = conf.make_model_conf().make_model()
            self.ema_model = copy.deepcopy(self.model)
            self.model.requires_grad_(False)
            self.ema_model.requires_grad_(False)
            self.ema_model.eval()

            if conf.pretrain is not None:
                print(f'loading pretrain ... {conf.pretrain.name}')
                state = torch.load(conf.pretrain.path, map_location='cpu')
                print('step:', state['global_step'])
                self.load_state_dict(state['state_dict'], strict=False)

            # load the latent stats
            if conf.manipulate_znormalize:
                print('loading latent stats ...')
                state = torch.load(conf.latent_infer_path)
                self.conds = state['conds']
                self.register_buffer('conds_mean',
                                     state['conds_mean'][None, :])
                self.register_buffer('conds_std', state['conds_std'][None, :])
            else:
                self.conds_mean = None
                self.conds_std = None

        if conf.manipulate_mode == 'texture':
            num_cls = len(TextureAttrDataset.id_to_cls)
        elif conf.manipulate_mode == 'tcga_crc_msi':
            num_cls = len(TcgaCrcMsiAttrDataset.id_to_cls)
        elif conf.manipulate_mode == 'tcga_crc_braf':
            num_cls = len(TCGACRCBRAFAttrDataset.id_to_cls)
        elif conf.manipulate_mode == 'brain':
            num_cls = len(BrainAttrDataset.id_to_cls)
        elif conf.manipulate_mode == 'pancancer':
            num_cls = len(PanCancerClsDataset.id_to_cls)
        elif conf.manipulate_mode == 'lung':
            num_cls = len(LungClsDataset.id_to_cls)
        elif conf.manipulate_mode == 'liver_cancer_types':
            num_cls = len(LiverCancerTypesClsDataset.id_to_cls)
        else:
            raise NotImplementedError()

        # classifier
        if conf.train_mode == TrainMode.manipulate:
            if conf.linear: 
                self.classifier = nn.Linear(conf.style_ch, num_cls)
            else:
                self.classifier = nn.Sequential(
                    nn.Linear(conf.style_ch, 512),
                    #nn.BatchNorm1d(512),
                    nn.SiLU(), # supposedly, cool kids use the SiLU nowadays: https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html 
                    nn.Dropout(0.5),
                    nn.Linear(512, num_cls)
                )
        else:
            raise NotImplementedError()

        self.ema_classifier = copy.deepcopy(self.classifier)

    def state_dict(self, *args, **kwargs):
        # don't save the base model
        out = {}
        for k, v in super().state_dict(*args, **kwargs).items():
            if k.startswith('model.'):
                pass
            elif k.startswith('ema_model.'):
                pass
            else:
                out[k] = v
        return out

    def load_state_dict(self, state_dict, strict: bool = None):
        if self.conf.train_mode == TrainMode.manipulate:
            # change the default strict => False
            if strict is None:
                strict = False
        else:
            if strict is None:
                strict = True
        return super().load_state_dict(state_dict, strict=strict)

    def normalize(self, cond):
        cond = (cond - self.conds_mean.to(self.device)) / self.conds_std.to(
            self.device)
        return cond

    def denormalize(self, cond):
        cond = (cond * self.conds_std.to(self.device)) + self.conds_mean.to(
            self.device)
        return cond

    def load_dataset(self):
        if self.conf.manipulate_mode == 'texture':
            return TextureAttrDataset(do_augment=True)
        if self.conf.manipulate_mode == 'tcga_crc_msi':
            return TcgaCrcMsiAttrDataset(do_augment=True)
        if self.conf.manipulate_mode == 'tcga_crc_braf':
            return TCGACRCBRAFAttrDataset(data_paths['tcga_crc_braf'],
                                    data_paths['tcga_crc_anno_braf'],
                                    do_augment=True)
        if self.conf.manipulate_mode == 'brain':
            return BrainAttrDataset(do_augment=True)
        if self.conf.manipulate_mode == 'lung':
            return LungClsDataset(do_augment=True)
        if self.conf.manipulate_mode == 'pancancer':
            return PanCancerClsDataset(do_augment=True)
        if self.conf.manipulate_mode == 'liver_cancer_types':
            return LiverCancerTypesClsDataset(do_augment=True)
        else:
            raise NotImplementedError()
        
        

    def setup(self, stage=None) -> None:
        ##############################################
        # NEED TO SET THE SEED SEPARATELY HERE
        if self.conf.seed is not None:
            seed = self.conf.seed * get_world_size() + self.global_rank
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            print('local seed:', seed)
        ##############################################

        self.train_data = self.load_dataset()


    def train_dataloader(self):
        # make sure to use the fraction of batch size
        # the batch size is global!
        conf = self.conf.clone()
        conf.batch_size = self.batch_size

        valid_indices = self.get_valid_indices(self.train_data)
        random.shuffle(valid_indices)
        sampler = CustomSampler(self.train_data, valid_indices)  # works just for nondistributed training

        if isinstance(self.train_data, list):
            dataloader = []
            for each in self.train_data:
                dataloader.append(
                    conf.make_loader(each, shuffle=False, drop_last=True, sampler=sampler))
            dataloader = ZipLoader(dataloader)
        else:
            dataloader = conf.make_loader(self.train_data,
                                          shuffle=True,
                                          drop_last=True,
                                          sampler=sampler)
        return dataloader

    def get_valid_indices(self, dataset):
        """
        Filters the dataset based on the valid filenames from the attr_path file.
        Returns the valid indices of the LMDB dataset.
        """
        valid_indices = []
        with lmdb.open(self.train_data.lmdb_path, readonly=True) as env:
            with env.begin(write=False) as txn:
                length_value = txn.get("length".encode("utf-8"))
                if length_value is not None:
                    length = int(length_value.decode("utf-8"))
                print(f"Length of LMDB dataset: {length}")
                for idx in tqdm(range(length)):
                    filename_key = f"filename_{str(idx).zfill(5)}".encode("utf-8")
                    filename = txn.get(filename_key).decode("utf-8")
                    fname = filename.split("/")[-2] + "/" + filename.split("/")[-1]
                    if fname in dataset.valid_fnames:
                        valid_indices.append(idx)
        print(f"Number of respective classes images found: {len(valid_indices)}")
        return valid_indices

    @property
    def batch_size(self):
        ws = get_world_size()
        assert self.conf.batch_size % ws == 0
        return self.conf.batch_size // ws

    def training_step(self, batch, batch_idx):
        self.ema_model: BeatGANsAutoencModel
        if isinstance(batch, tuple):
            a, b = batch
            imgs = torch.cat([a['img'], b['img']])
            labels = torch.cat([a['labels'], b['labels']])
        else:
            imgs = batch['img']
            labels = batch['labels']
            # fname = batch['filename']

        # print(fname)

        if self.conf.train_mode == TrainMode.manipulate:
            self.ema_model.eval()
            with torch.no_grad():
                # (n, c)
                cond = self.ema_model.encoder(imgs)

            if self.conf.manipulate_znormalize:
                cond = self.normalize(cond)

            # (n, cls)
            pred = self.classifier.forward(cond)
            # print(f"pred: {pred}")
            pred_ema = self.ema_classifier.forward(cond)
        else:
            raise NotImplementedError()

        gt = torch.where(labels > 0,
                            torch.ones_like(labels).float(),
                            torch.zeros_like(labels).float())
        # print(f"GT: {gt}")
        if self.conf.manipulate_loss == ManipulateLossType.bce:
            loss = F.binary_cross_entropy_with_logits(pred, gt)
            if pred_ema is not None:
                loss_ema = F.binary_cross_entropy_with_logits(pred_ema, gt)
        elif self.conf.manipulate_loss == ManipulateLossType.mse:
            loss = F.mse_loss(pred, gt)
            if pred_ema is not None:
                loss_ema = F.mse_loss(pred_ema, gt)
        else:
            raise NotImplementedError()

        self.log('loss', loss)
        # print(f"loss: {loss}")
        self.log('loss_ema', loss_ema)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        ema(self.classifier, self.ema_classifier, self.conf.ema_decay)

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.classifier.parameters(),
                                 lr=self.conf.lr,
                                 weight_decay=self.conf.weight_decay)
        return optim
    
    def test_step(self, batch, dataloader=None):

        data = batch['img']
        label = batch['labels']
        fname = batch['filename']
        print(f"Processing file: {fname}")
        
        data, label = data.cuda().float(), label.cuda().float(),
        # print(f"GT label: {label}")

        # encode the image into the latent space of the diffusion model
        latent = self.model.encoder(data)
        latent = self.normalize(latent)

        output = self.classifier.forward(latent)
        # print(f"Output: {output}")                
        # pred = torch.sigmoid(output)
        pred = torch.softmax(output)
        # print(f"Pred: {pred}")    
        
        # label_list.append(label.cpu().numpy())
        # pred_list.append(pred.cpu().numpy())
        # fnames.append(fname[0])


def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(target_dict[key].data * decay +
                                    source_dict[key].data * (1 - decay))


def train_cls(conf: TrainConfig, gpus):
    print('conf:', conf.name)
    model = ClsModel(conf)

    if not os.path.exists(conf.logdir):
        os.makedirs(conf.logdir)
    checkpoint = ModelCheckpoint(
        dirpath=f'{conf.logdir}',
        save_last=True,
        save_top_k=1,
        monitor = 'loss',
        filename='best_model-epoch={epoch:02d}-step={step}-loss={loss:.4f}',
        every_n_train_steps=conf.save_every_samples // conf.batch_size_effective,
    )
    checkpoint_path = f'{conf.logdir}/last.ckpt'
    if os.path.exists(checkpoint_path):
        resume = checkpoint_path
    else:
        if conf.continue_from is not None:
            # continue from a checkpoint
            resume = conf.continue_from.path
        else:
            resume = None

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=conf.logdir,
                                             name=None,
                                             version='')

    plugins = []
    if len(gpus) == 1:
        accelerator = "gpu"
        strategy = "auto"
    elif len(gpus) > 1:
        """ # older pytorch-lightning version (e.g. 2.0.6)
        accelerator = 'ddp'
        from pytorch_lightning.plugins import DDPPlugin
        # important for working with gradient checkpoint
        plugins.append(DDPPlugin(find_unused_parameters=True))
        """
        from pytorch_lightning.strategies import DDPStrategy     # pytorch-lightning version 2.1.1
        strategy = DDPStrategy(find_unused_parameters=True)
        # strategy = 'ddp'
        accelerator = 'gpu'
    else:
        accelerator = "cpu"

    trainer = pl.Trainer(
        max_steps=conf.total_samples // conf.batch_size_effective,
        # resume_from_checkpoint=resume,  # older pytorch-lightning version (e.g. 2.0.6)
        # gpus=gpus,                      # older pytorch-lightning version (e.g. 2.0.6)
        devices=gpus,                     # only for newer pytorch-lightning versions (2.1.1)
        strategy=strategy,                # only for newer pytorch-lightning versions (2.1.1)
        accelerator=accelerator,
        precision="16-mixed" if conf.fp16 else 32,
        callbacks=[
            checkpoint,
        ],
        # replace_sampler_ddp=True,
        logger=tb_logger,
        accumulate_grad_batches=conf.accum_batches,
        plugins=plugins,
    )
    trainer.fit(model, 
                ckpt_path=resume, # only for newer pytorch-lightning versions (2.1.1)
                )

    """
    gt_table_dir = f"{ws_path}/mopadi/datasets/brain_gt_test.csv"
    images_dir = f"{ws_path}/data/brain"

    test_dataset = BrainDataset(images_dir=images_dir,
                                path_to_gt_table=gt_table_dir,
                                transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ]))
    """
    
    # test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
    
    # trainer.test(model=model, dataloaders=test_dataloader)
    