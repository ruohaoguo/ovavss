from .text_prompt import get_predefined_templates
from .adapter import ClipAdapter, BgClipAdapter
from .mask_adapted_adapter import AdaptedClipAdapter, BgAdaptedClipAdapter
from .side_adapter import SideAdapter

ADAPTER_REGISTER = {x.__name__: x for x in [ClipAdapter, BgClipAdapter, AdaptedClipAdapter, BgAdaptedClipAdapter, SideAdapter]}


def build_clip_adapter(cfg):
    if cfg.NAME in ['ClipAdapter', 'BgClipAdapter']:
        return ADAPTER_REGISTER[cfg.NAME](cfg.CLIP_MODEL_NAME, text_templates=get_predefined_templates(cfg.PROMPT_NAME))
    elif cfg.NAME in ['AdaptedClipAdapter', 'BgAdaptedClipAdapter']:
        return ADAPTER_REGISTER[cfg.NAME](cfg.CLIP_MODEL_NAME, mask_prompt_depth=cfg.MASK_PROMPT_DEPTH, mask_prompt_fwd=cfg.MASK_PROMPT_FWD, text_templates=get_predefined_templates(cfg.PROMPT_NAME))
    elif cfg.NAME in ['SideAdapter']:
        return ADAPTER_REGISTER[cfg.NAME]()


def default_clip_adapter():
    return ClipAdapter('ViT-B/16', get_predefined_templates('vild'))


def default_bg_clip_adapter():
    return BgClipAdapter('ViT-B/16', get_predefined_templates('vild'))

