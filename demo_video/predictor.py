import atexit
import bisect
import multiprocessing as mp
import numpy as np
import torch

from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import ColorMode


class VisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.parallel = parallel
        if parallel:
            num_gpu = torch.cuda.device_count()
            self.predictor = AsyncPredictor(cfg, num_gpus=num_gpu)
        else:
            self.predictor = VideoPredictor(cfg)

    def run_on_video_avsbench(self, frames, file_names, audios, threshold):
        frames_audios = [frames, file_names, audios]
        predictions_all = self.predictor(frames_audios)

        pred_scores = predictions_all["pred_scores"]
        pred_labels = predictions_all["pred_labels"]
        pred_masks = predictions_all["pred_masks"]

        # select high-score masks
        pred_scores_p = []
        pred_labels_p = []
        pred_masks_p = []
        for p in range(len(pred_scores)):
            if pred_scores[p] > threshold:
                pred_scores_p.append(pred_scores[p])
                pred_labels_p.append(pred_labels[p])
                pred_masks_p.append(pred_masks[p])

        # sort
        sorted_indices = sorted(range(len(pred_scores_p)), key=lambda i: pred_scores_p[i], reverse=True)
        sorted_labels = [pred_labels_p[i] for i in sorted_indices]
        sorted_masks = [pred_masks_p[i] for i in sorted_indices]

        if sorted_masks != []:
            predictions_all_masks = torch.stack(sorted_masks, dim=0)
            predictions_all_masks = predictions_all_masks.permute(1, 0, 2, 3)
            predictions_all_labels = torch.tensor(sorted_labels)
        else:
            predictions_all_masks = torch.zeros((len(frames), 1, frames[0].shape[0], frames[0].shape[1]), device="cuda:0")
            predictions_all_labels = torch.tensor(-1, device="cuda:0").unsqueeze(dim=0)

        return predictions_all_labels, predictions_all_masks


class VideoPredictor(DefaultPredictor):
    """
    Create a simple end-to-end predictor with the given config that runs on
    single device for a single input image.
    Compared to using the model directly, this class does the following additions:
    1. Load checkpoint from `cfg.MODEL.WEIGHTS`.
    2. Always take BGR image as the input and apply conversion defined by `cfg.INPUT.FORMAT`.
    3. Apply resizing defined by `cfg.INPUT.{MIN,MAX}_SIZE_TEST`.
    4. Take one input image and produce a single output, instead of a batch.
    If you'd like to do anything more fancy, please refer to its source code
    as examples to build and use the model manually.
    Attributes:
        metadata (Metadata): the metadata of the underlying dataset, obtained from
            cfg.DATASETS.TEST.
    Examples:
    ::
        pred = DefaultPredictor(cfg)
        inputs = cv2.imread("input.jpg")
        outputs = pred(inputs)
    """
    def __call__(self, frames_audios):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).
        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            input_frames = []
            frames = frames_audios[0]
            file_names = frames_audios[1]
            audios = frames_audios[2]
            for original_image in frames:
                # Apply pre-processing to image.
                if self.input_format == "RGB":
                    # whether the model expects BGR inputs or RGB
                    original_image = original_image[:, :, ::-1]
                height, width = original_image.shape[:2]
                image = self.aug.get_transform(original_image).apply_image(original_image)
                image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
                input_frames.append(image)

            inputs = {"image": input_frames, "height": height, "width": width, "file_names":file_names, "wav_data": audios}

            # from thop import profile
            # flops, params = profile(self.model, inputs=([inputs],))
            # flops_gb = flops * 4 / (1024 ** 3)
            # params_mb = params * 4 / (1024 ** 2)
            # print(f"FLOPs: {flops_gb}")
            # print(f"Params: {params_mb}")

            predictions = self.model([inputs])
            return predictions


class AsyncPredictor:
    """
    A predictor that runs the model asynchronously, possibly on >1 GPUs.
    Because rendering the visualization takes considerably amount of time,
    this helps improve throughput when rendering videos.
    """

    class _StopToken:
        pass

    class _PredictWorker(mp.Process):
        def __init__(self, cfg, task_queue, result_queue):
            self.cfg = cfg
            self.task_queue = task_queue
            self.result_queue = result_queue
            super().__init__()

        def run(self):
            predictor = VideoPredictor(self.cfg)

            while True:
                task = self.task_queue.get()
                if isinstance(task, AsyncPredictor._StopToken):
                    break
                idx, data = task
                result = predictor(data)
                self.result_queue.put((idx, result))

    def __init__(self, cfg, num_gpus: int = 1):
        """
        Args:
            cfg (CfgNode):
            num_gpus (int): if 0, will run on CPU
        """
        num_workers = max(num_gpus, 1)
        self.task_queue = mp.Queue(maxsize=num_workers * 3)
        self.result_queue = mp.Queue(maxsize=num_workers * 3)
        self.procs = []
        for gpuid in range(max(num_gpus, 1)):
            cfg = cfg.clone()
            cfg.defrost()
            cfg.MODEL.DEVICE = "cuda:{}".format(gpuid) if num_gpus > 0 else "cpu"
            self.procs.append(
                AsyncPredictor._PredictWorker(cfg, self.task_queue, self.result_queue)
            )

        self.put_idx = 0
        self.get_idx = 0
        self.result_rank = []
        self.result_data = []

        for p in self.procs:
            p.start()
        atexit.register(self.shutdown)

    def put(self, image):
        self.put_idx += 1
        self.task_queue.put((self.put_idx, image))

    def get(self):
        self.get_idx += 1  # the index needed for this request
        if len(self.result_rank) and self.result_rank[0] == self.get_idx:
            res = self.result_data[0]
            del self.result_data[0], self.result_rank[0]
            return res

        while True:
            # make sure the results are returned in the correct order
            idx, res = self.result_queue.get()
            if idx == self.get_idx:
                return res
            insert = bisect.bisect(self.result_rank, idx)
            self.result_rank.insert(insert, idx)
            self.result_data.insert(insert, res)

    def __len__(self):
        return self.put_idx - self.get_idx

    def __call__(self, image):
        self.put(image)
        return self.get()

    def shutdown(self):
        for _ in self.procs:
            self.task_queue.put(AsyncPredictor._StopToken())

    @property
    def default_buffer_size(self):
        return len(self.procs) * 5
