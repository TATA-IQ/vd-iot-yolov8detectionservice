import yaml

from yaml.loader import SafeLoader


class Config:
    """
    Load the configuration file
    """
    def yamlconfig(path):
        """
        Load the configuration file with minio, sftp and apis
        Args:
            path (str): path of the config file
        """
        with open(path, "r") as f:
            data = list(yaml.load_all(f, Loader=SafeLoader))
            print(data)
        return data

    def yamlModel(path):
        """
        Load the model configuration file
        Args:
            path (str): path of the model config file
        """
        with open(path, "r") as f:
            data = list(yaml.load_all(f, Loader=SafeLoader))
            print(data)
        return data