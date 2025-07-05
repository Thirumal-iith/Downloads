import os
from urllib.parse import urlparse
from model_registry import ModelRegistry, utils

def infer_model_format(uri):
    """Infer model format from file extension."""
    ext = os.path.splitext(uri)[1].lower()
    return {
        ".onnx": "onnx",
        ".pkl": "sklearn",
        ".pt": "pytorch",
        ".joblib": "sklearn",
        ".h5": "keras",
        ".sav": "sklearn"
    }.get(ext, "unknown")

def is_huggingface_model(uri_or_repo):
    """Detect if string looks like a Hugging Face model repo."""
    return (
        isinstance(uri_or_repo, str)
        and "/" in uri_or_repo
        and not uri_or_repo.startswith(("http", "s3://", "minio://", "/"))
    )

def register_model_allinone(
    registry_url,
    model_name,
    uri_or_repo,
    author=None,
    storage_key=None,
    model_format=None,
    description=None,
    version=None,
    metadata=None
):
    """
    Universal model registration for MinIO, S3, HF, public URLs, and local files.
    """
    # Step 1: Initialize client
    registry = ModelRegistry(registry_url, author=author)

    # Step 2: Register Hugging Face model (repo ID style)
    if is_huggingface_model(uri_or_repo):
        return registry.register_huggingface_model(
            model_name,
            repo_id=uri_or_repo,
            description=description,
            version=version,
            metadata=metadata or {}
        )

    # Step 3: Auto-infer model format if not given
    if not model_format:
        model_format = infer_model_format(uri_or_repo)

    # Step 4: Convert MinIO/S3 URIs using storage key
    if uri_or_repo.startswith("minio://") and storage_key:
        uri_or_repo = utils.s3_uri_from(uri_or_repo.replace("minio://", ""), storage_key)
    elif uri_or_repo.startswith("s3://") and storage_key:
        uri_or_repo = utils.s3_uri_from(uri_or_repo.replace("s3://", ""), storage_key)

    # Step 5: Register model
    return registry.register_model(
        model_name,
        uri=uri_or_repo,
        model_format=model_format,
        description=description,
        version=version,
        storage_key=storage_key,
        metadata=metadata or {}
    )


register_model_allinone(
    registry_url="https://your-registry.com",
    model_name="my-cool-model",
    uri_or_repo="minio://mlpipeline/models/my-model.onnx",
    storage_key="mlpipeline",
    version = 1,
    metadata={
        "task": "classification",
        "framework": "onnx",
        "team": "mlops",
        "dataset": "imagenet"
    }
)
