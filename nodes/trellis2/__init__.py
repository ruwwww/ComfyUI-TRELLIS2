import os

# Suppress verbose HTTP request logs from huggingface_hub/httpx
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

log = logging.getLogger("trellis2")


def _comfy_tqdm():
    """tqdm that shows download progress in ComfyUI's UI."""
    try:
        import comfy.utils
        import tqdm as _tqdm_mod
    except ImportError:
        return None
    holder = {"pbar": None, "total": 0, "done": 0}
    class _T(_tqdm_mod.tqdm):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            if self.total and self.total > 0 and holder["pbar"] is None:
                holder["total"] = self.total
                holder["done"] = 0
                holder["pbar"] = comfy.utils.ProgressBar(self.total)
        def update(self, n=1):
            ret = super().update(n)
            if n and holder["pbar"] and holder["total"] > 0:
                holder["done"] = min(holder["done"] + n, holder["total"])
                holder["pbar"].update_absolute(holder["done"], holder["total"])
            return ret
    return _T


# Model class lookup table: name -> (module, class_name)
_MODEL_CLASSES = {
    # Sparse Structure
    'SparseStructureEncoder': ('.vae', 'SparseStructureEncoder'),
    'SparseStructureDecoder': ('.vae', 'SparseStructureDecoder'),
    'SparseStructureFlowModel': ('.model', 'SparseStructureFlowModel'),
    # SLat Generation
    'SLatFlowModel': ('.model', 'SLatFlowModel'),
    'ElasticSLatFlowModel': ('.model', 'ElasticSLatFlowModel'),
    # SC-VAEs
    'SparseUnetVaeEncoder': ('.vae', 'SparseUnetVaeEncoder'),
    'SparseUnetVaeDecoder': ('.vae', 'SparseUnetVaeDecoder'),
    'FlexiDualGridVaeEncoder': ('.vae', 'FlexiDualGridVaeEncoder'),
    'FlexiDualGridVaeDecoder': ('.vae', 'FlexiDualGridVaeDecoder'),
}


def _get_trellis2_models_dir():
    """Get the ComfyUI/models/trellis2 directory."""
    try:
        import folder_paths
        models_dir = os.path.join(folder_paths.models_dir, "trellis2")
    except ImportError:
        models_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "models", "trellis2")
    os.makedirs(models_dir, exist_ok=True)
    return models_dir


def _get_model_class(name: str):
    """Resolve a model class name to the actual class."""
    import importlib
    if name not in _MODEL_CLASSES:
        raise ValueError(f"Unknown model class: {name}")
    module_path, class_name = _MODEL_CLASSES[name]
    module = importlib.import_module(module_path, __name__)
    return getattr(module, class_name)


def from_pretrained(path: str, disk_offload_manager=None, model_key: str = None, device=None, **kwargs):
    """
    Load a model from a pretrained checkpoint.

    Args:
        path: The path to the checkpoint. Can be either local path or a Hugging Face model name.
              NOTE: config file and model file should take the name f'{path}.json' and f'{path}.safetensors' respectively.
        disk_offload_manager: Optional DiskOffloadManager for RAM-efficient loading.
        model_key: Optional key to identify this model in the disk_offload_manager.
        **kwargs: Additional arguments for the model constructor.
    """
    import json
    import torch
    import comfy.utils

    # Check if it's a direct local path
    is_local = os.path.exists(f"{path}.json") and os.path.exists(f"{path}.safetensors")

    if is_local:
        config_file = f"{path}.json"
        model_file = f"{path}.safetensors"
    else:
        # Parse HuggingFace path
        path_parts = path.split('/')
        repo_id = f'{path_parts[0]}/{path_parts[1]}'
        model_name = '/'.join(path_parts[2:])

        # Check if cached in ComfyUI/models/trellis2
        models_dir = _get_trellis2_models_dir()
        local_config = os.path.join(models_dir, f"{model_name}.json")
        local_weights = os.path.join(models_dir, f"{model_name}.safetensors")

        # Create subdirectories if needed
        os.makedirs(os.path.dirname(local_config), exist_ok=True)

        if os.path.exists(local_config) and os.path.exists(local_weights):
            log.info(f"Loading {model_name} from local cache...")
            config_file = local_config
            model_file = local_weights
        else:
            # Download directly to models folder (no intermediate HF cache)
            from huggingface_hub import hf_hub_download
            log.info(f"Downloading {model_name} config...")
            hf_hub_download(repo_id, f"{model_name}.json", local_dir=models_dir, tqdm_class=_comfy_tqdm())
            log.info(f"Downloading {model_name} weights (this may take a while)...")
            hf_hub_download(repo_id, f"{model_name}.safetensors", local_dir=models_dir, tqdm_class=_comfy_tqdm())
            config_file = local_config
            model_file = local_weights

    with open(config_file, 'r') as f:
        config = json.load(f)

    # Auto-detect device
    if device is None:
        import comfy.model_management
        device = str(comfy.model_management.get_torch_device())

    sd = comfy.utils.load_torch_file(model_file, device=torch.device(device))
    quant_config = comfy.utils.detect_layer_quantization(sd, "")

    # Build model on meta device (zero memory, no random init)
    model_class = _get_model_class(config['name'])
    op_kwargs = {}
    if quant_config:
        import inspect
        import comfy.ops
        from .ops_sparse import mixed_precision_sparse_ops
        compute_dtype = torch.bfloat16
        disabled = comfy.ops.get_disabled_quant_formats(torch.device(device))
        sig = inspect.signature(model_class.__init__)
        if "operations" in sig.parameters and "operations" not in kwargs:
            op_kwargs["operations"] = comfy.ops.mixed_precision_ops(
                quant_config=quant_config, compute_dtype=compute_dtype, disabled=disabled
            )
        if "sparse_operations" in sig.parameters and "sparse_operations" not in kwargs:
            op_kwargs["sparse_operations"] = mixed_precision_sparse_ops(
                quant_config=quant_config, compute_dtype=compute_dtype, disabled=disabled
            )

    log.info(f"Building model: {config['name']} (meta device)...")
    with torch.device("meta"):
        model = model_class(**config['args'], **op_kwargs, **kwargs)
    log.info(f"Loading weights directly to {device}...")
    model.load_state_dict(sd, strict=False, assign=True)
    del sd
    weight_dtype = next(model.parameters()).dtype
    import sys; print(f"[TRELLIS2] Model {config['name']} loaded: weights={weight_dtype}, quant={bool(quant_config)}, device={device}", file=sys.stderr)

    # Reinitialize any buffers left on meta device after assign=True loading
    for name, buf in model.named_buffers():
        if buf.device.type == "meta":
            parts = name.split(".")
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            parent._buffers[parts[-1]] = torch.zeros_like(buf, device=device)

    # Recompute derived buffers (e.g., RoPE phases).
    if hasattr(model, '_post_load'):
        model._post_load(torch.device(device))
        log.info(f"Recomputed derived buffers for {config['name']}")

    # Register with disk offload manager if provided
    if disk_offload_manager is not None:
        if model_key is None:
            raise ValueError("model_key is required when disk_offload_manager is provided")
        disk_offload_manager.register(model_key, model_file)

    return model
