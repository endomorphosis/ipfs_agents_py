from transformers import Tool
from huggingface_hub import list_models
from ipfs_kit import ipfs_kit
from orbitdb_kit import orbitdb_kit 
from ipfs_model_manager import ipfs_model_manager as model_manager 
from ipfs_model_manager import load_config, load_collection
from ipfs_model_manager import list_ipfs_models, list_s3_models, list_local_models, ModelManager

class HFModelDownloadsTool(Tool):
    name = "model_download_counter"
    description = (
        "This is a tool that returns the most downloaded model of a given task on the Hugging Face Hub. "
        "It takes the name of the category (such as text-classification, depth-estimation, etc), and "
        "returns the name of the checkpoint."
    )

    inputs = ["text"]
    outputs = ["text"]

    def __call__(self, task: str):
        model = next(iter(list_models(filter=task, sort="downloads", direction=-1)))
        return model.id

class IPFSModelDownloadsTool(Tool):
    name = "model_download_registry"
    description = ("This is a tool that returns the the models that are available on IPFS. ")
    inputs = ["text"]
    outputs = ["text"]

    def __call__(self, task: str):
        model = next(iter(list_models(filter=task, sort="downloads", direction=-1)))
        return model.id

class S3ModelDownloadsTool(Tool):
    name = "model_download_registry"
    description = ("This is a tool that returns the the models that are available on S3. ")
    inputs = ["text"]
    outputs = ["text"]

    def __call__(self, task: str):
        model = next(iter(list_models(filter=task, sort="downloads", direction=-1)))
        return model.id
    
class LocalModelDownloadsTool(Tool):
    name = "model_download_registry"
    description = ("This is a tool that returns the the models that are available locally. ")
    inputs = ["text"]
    outputs = ["text"]

    def __call__(self, task: str):
        model = next(iter(list_models(filter=task, sort="downloads", direction=-1)))
        return model.id

if __name__ == "__main__":
    config = load_config()
    collection = load_collection()
    models = ModelManager()
    ready = models.ready()
    models.import_config(config)
    models.import_collection(collection)
    models.state()