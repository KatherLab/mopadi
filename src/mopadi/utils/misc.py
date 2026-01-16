# Code sourced from the Official implementation of Diffusion Autoencoders by Konpat Preechakul
# Original Source: https://github.com/phizaz/diffae
# License: MIT

from io import BytesIO
from PIL import Image

import torch

from contextlib import contextmanager
from torch.utils.data import Dataset
from multiprocessing import Process, Queue
import os
import shutil

from mopadi.configs.config import *
    

def render_condition(conf: TrainConfig,
                     model: BeatGANsAutoencModel,
                     x_T,
                     sampler: Sampler,
                     cond):
    """
    Generate samples conditioned on extracted features.
    """
    model.eval()
    assert conf.model_type.has_autoenc()
    with torch.no_grad():
        return sampler.sample(model=model, noise=x_T, cond=cond)



def convert(x, format, quality=100):
    # to prevent locking!
    torch.set_num_threads(1)

    buffer = BytesIO()
    x = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0)
    x = x.to(torch.uint8)
    x = x.numpy()
    img = Image.fromarray(x)
    img.save(buffer, format=format, quality=quality)
    val = buffer.getvalue()
    return val


@contextmanager
def nullcontext():
    yield


class _WriterWroker(Process):
    def __init__(self, path, format, quality, zfill, q):
        super().__init__()
        if os.path.exists(path):
            shutil.rmtree(path)

        self.path = path
        self.format = format
        self.quality = quality
        self.zfill = zfill
        self.q = q
        self.i = 0

    def run(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        with lmdb.open(self.path, map_size=1024**4, readahead=False) as env:
            while True:
                job = self.q.get()
                if job is None:
                    break
                with env.begin(write=True) as txn:
                    for x in job:
                        key = f"{str(self.i).zfill(self.zfill)}".encode(
                            "utf-8")
                        x = convert(x, self.format, self.quality)
                        txn.put(key, x)
                        self.i += 1

            with env.begin(write=True) as txn:
                txn.put("length".encode("utf-8"), str(self.i).encode("utf-8"))


class LMDBImageWriter:
    """
    Needed only for inferring features for training the latent DPM and linear classifier.
    TODO: get rid of this approach and do it the same way as for MIL classifier.
    """
    def __init__(self, path, format='webp', quality=100, zfill=7) -> None:
        self.path = path
        self.format = format
        self.quality = quality
        self.zfill = zfill
        self.queue = None
        self.worker = None

    def __enter__(self):
        self.queue = Queue(maxsize=3)
        self.worker = _WriterWroker(self.path, self.format, self.quality,
                                    self.zfill, self.queue)
        self.worker.start()

    def put_images(self, tensor):
        """
        Args:
            tensor: (n, c, h, w) [0-1] tensor
        """
        self.queue.put(tensor.cpu())

    def __exit__(self, *args, **kwargs):
        self.queue.put(None)
        self.queue.close()
        self.worker.join()
