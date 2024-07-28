from easydict import EasyDict as edict

vgg_cfg = edict()
vgg_cfg.TRAIN = edict()

vgg_cfg.TRAIN.FREEZE_AUDIO_EXTRACTOR = True
vgg_cfg.TRAIN.PRETRAINED_VGGISH_MODEL_PATH = "./pre_models/vggish-10086976.pth"
vgg_cfg.TRAIN.PREPROCESS_AUDIO_TO_LOG_MEL = True
vgg_cfg.TRAIN.POSTPROCESS_LOG_MEL_WITH_PCA = False
vgg_cfg.TRAIN.PRETRAINED_PCA_PARAMS_PATH = "./pre_models/vggish_pca_params-970ea276.pth"
