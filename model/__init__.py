from .film import FiLM

from utils.io import get_checkpoint


def get_model(path):
    ckpt = get_checkpoint(path)
    model = FiLM.load_from_checkpoint(ckpt)
    return model