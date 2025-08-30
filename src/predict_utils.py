from pathlib import Path
import sys
import logging
import torch
from typing import Any, Dict, Optional


def setup_project_env(project_root: Optional[Path] = None, log_level: int = logging.INFO):
    if project_root is None:
        project_root = Path.cwd().parents[1].resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    logging.basicConfig(level=log_level, format='[%(levelname)s] %(message)s')
    logger = logging.getLogger(__name__)
    torch.cuda.empty_cache()
    logger.info("Environment ready. Project root: %s", project_root)
    return logger, project_root


def log_torch_info(logger: logging.Logger):
    print("Torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Device:", torch.cuda.get_device_name(0))
    logger.info("Torch: %s", torch.__version__)
    logger.info("CUDA available: %s", torch.cuda.is_available())
    if torch.cuda.is_available():
        logger.info("Device: %s", torch.cuda.get_device_name(0))


def build_weights_path(workspace_root: str, *parts: str) -> str:
    return str(Path(workspace_root).joinpath(*parts))


def load_rtdetr(weights_path: str, logger: logging.Logger) -> Any:
    from ultralytics import RTDETR
    logger.info("Weights path: %s", weights_path)
    model = RTDETR(weights_path)
    logger.info("RTDETR model instantiated with weights: %s", weights_path)
    return model


def load_yolo(weights_path: str, logger: logging.Logger) -> Any:
    from ultralytics import YOLO
    logger.info("Weights path: %s", weights_path)
    model = YOLO(weights_path)
    logger.info("YOLO model instantiated with weights: %s", weights_path)
    return model


def make_predict_args(
    source: str,
    imgsz: int = 960,
    conf: float = 0.70,
    device: int = 0,
    half: bool = True,
    batch: int = 4,
    workers: int = 0,
    stream: bool = False,
    save: bool = True,
    save_txt: bool = True,
    save_conf: bool = True,
    project: Optional[str] = None,
    name: str = "exp",
) -> Dict[str, Any]:
    return dict(
        source=source,
        imgsz=imgsz,
        conf=conf,
        device=device,
        half=half,
        batch=batch,
        workers=workers,
        stream=stream,
        save=save,
        save_txt=save_txt,
        save_conf=save_conf,
        project=project,
        name=name,
    )


def predict_with_oom_retry(model: Any, args: Dict[str, Any], logger: logging.Logger, fallback_imgsz: int = 768, fallback_batch: int = 1) -> Any:
    try:
        logger.info("Calling model.predict")
        return model.predict(**args)
    except RuntimeError as e:
        msg = str(e)
        print("RuntimeError:", msg)
        logger.error("RuntimeError during prediction: %s", msg)
        if "CUDA out of memory" in msg or "CUDNN_STATUS_ALLOC_FAILED" in msg:
            print("\nOOM detected. Retrying with smaller settings...")
            logger.warning("OOM detected; retrying with imgsz=%s, batch=%s, half=True", fallback_imgsz, fallback_batch)
            torch.cuda.empty_cache()
            args.update(dict(imgsz=fallback_imgsz, batch=fallback_batch, half=True))
            logger.info("Retrying model.predict with args: %s", args)
            return model.predict(**args)
        else:
            raise
