import os
import json
import argparse
from urllib.parse import urlparse
from model_registry import ModelRegistry, utils

def infer_model_format(uri):
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
    return isinstance(uri_or_repo, str) and "/" in uri_or_repo and not uri_or_repo.startswith(("http", "s3://", "minio://", "/"))

def register_model_allinone(model_name, uri_or_repo, user_token, author, storage_key, model_format, description, version, metadata):
    if not user_token:
        raise ValueError("User token is required for secure connection.")

    base_url = "http://model-registry-service.kubeflow.svc.cluster.local:8080"
    is_secure = False

    registry = ModelRegistry(
        server_address=base_url,
        author=author or "Anonymous",
        is_secure=is_secure,
        user_token=user_token
    )

    version_int = int(version)

    if is_huggingface_model(uri_or_repo):
        return registry.register_huggingface_model(
            model_name,
            repo_id=uri_or_repo,
            description=description,
            version=version_int,
            metadata=metadata
        )

    if not model_format:
        model_format = infer_model_format(uri_or_repo)
        if model_format == "unknown":
            raise ValueError(f"Unknown model format for URI: {uri_or_repo}")

    if uri_or_repo.startswith("minio://"):
        if not storage_key:
            raise ValueError("Storage key required for MinIO URIs.")
        uri_or_repo = utils.s3_uri_from(uri_or_repo.replace("minio://", ""), storage_key)
    elif uri_or_repo.startswith("s3://"):
        if not storage_key:
            raise ValueError("Storage key required for S3 URIs.")
        uri_or_repo = utils.s3_uri_from(uri_or_repo.replace("s3://", ""), storage_key)

    return registry.register_model(
        model_name,
        uri=uri_or_repo,
        description=description,
        version=version_int,
        model_format_name=model_format,
        model_format_version="1",
        storage_key=storage_key,
        metadata=metadata
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--uri_or_repo", required=True)
    parser.add_argument("--user_token", required=True)
    parser.add_argument("--author", default="")
    parser.add_argument("--storage_key", default="")
    parser.add_argument("--model_format", default="")
    parser.add_argument("--description", default="No description provided")
    parser.add_argument("--version", default="1")
    parser.add_argument("--metadata", default="{}")
    args = parser.parse_args()

    try:
        metadata = json.loads(args.metadata) if args.metadata else {}
    except Exception:
        metadata = {}

    result = register_model_allinone(
        args.model_name,
        args.uri_or_repo,
        args.user_token,
        args.author,
        args.storage_key,
        args.model_format,
        args.description,
        args.version,
        metadata
    )

    print("Model:", result.get("model") if isinstance(result, dict) else result.model)
    print("Model Version:", result.get("model_version") if isinstance(result, dict) else result.version)
    print("Model Artifacts:", result.get("model_artifacts") if isinstance(result, dict) else result.artifacts)
