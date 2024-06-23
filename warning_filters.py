import warnings

def suppress_specific_warnings():
    # Suppress PyTorch deprecation warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils._pytree")
    warnings.filterwarnings("ignore", category=UserWarning, message="torch.utils._pytree._register_pytree_node is deprecated")

    # Suppress Hugging Face Hub warnings
    warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.file_download")