
# setuptools error can be fixed with conda update --force conda

import mlflow, os, hydra, json
from omegaconf import DictConfig, OmegaConf
@hydra.main(config_name='config', config_path='config', version_base='2.5')
def go(config: DictConfig):

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]


    root_path = hydra.utils.get_original_cwd()
    if isinstance(config["main"]["execute_steps"], str):
        steps_to_execute = config["main"]["execute_steps"].split(",")
    else:
        steps_to_execute = config["main"]["execute_steps"]


    if "download" in steps_to_execute:
        # Download file and load in W&B
        _ = mlflow.run(
            f"{config['main']['download_repository']}", 
            version="main",
            entry_point="main",
            parameters={
                    "file_url": config["data"]["file_url"],
                    "artifact_name": config["data"]["raw_data_name"],#"raw_data.csv", 
                    "artifact_type": config["data"]["raw_data_type"], # "raw_data", 
                    "artifact_description": config["data"]["raw_data_description"]}) #"Raw file as downloaded" 


    if "basic_cleaning" in steps_to_execute:

        _ = mlflow.run(
            os.path.join(root_path, "basic_cleaning"),
            entry_point="main",
            parameters={
                "input_artifact": config["data"]["raw_data_artifact"], #"raw_data.csv:latest",
                "output_name": config["data"]["preprocessed_data"], #was artifact_name
                "output_type": "preprocessed_data", # was artifact_type
                "output_description": "Data with preprocessing applied", #was artifact description
                "min_price": config["data"]["min_price"],
                "max_price": config["data"]["max_price"]})

    if "check_data" in steps_to_execute:

        _ = mlflow.run(
            os.path.join(root_path, "check_data"),
            entry_point="main",
            parameters={
                "reference_artifact": config["data"]["reference_dataset"],
                "sample_artifact": config["data"]["preprocessed_data_latest"], # "preprocessed_data.csv:latest",
                "kl_threshold": config["data"]["kl_threshold"],
                "min_price": config["data"]["min_price"],
                "max_price": config["data"]["max_price"]})

  #  if "EDA" in steps_to_execute:
  #      _ = mlflow.run(
  #          os.path.join(root_path, "EDA"),
  #          entry_point="main")

    if "segregate" in steps_to_execute:
        _ = mlflow.run(
            os.path.join(root_path, "segregate"),
            entry_point="main",
            parameters={
                "input_artifact": config["data"]["preprocessed_data_latest"], #preprocessed_data.csv:latest",
                "artifact_root": "data",
                "artifact_type": "segregated_data",
                "test_size": config["data"]["test_size"],
                "stratify": config["data"]["stratify"]})

    if "random_forest" in steps_to_execute:

        # NOTE: we need to serialize the random forest configuration into JSON
        model_config=os.path.abspath("random_forest_config.yml")
        rf_config = os.path.abspath("rf_config.json")
        with open(rf_config, "w+") as fp:
            json.dump(dict(config["random_forest_pipeline"]["random_forest"].items()), fp)
        with open(model_config, "w+", encoding="utf-8") as fp:
            fp.write(OmegaConf.to_yaml(config["random_forest_pipeline"]))
        _ = mlflow.run(
            os.path.join(root_path, "random_forest"),
            entry_point="main", 
            parameters={
                "trainval_artifact": config["data"]["train_data"], #"data_train.csv:latest",
                "rf_config": rf_config,
                "output_artifact": config["random_forest_pipeline"]["output_artifact"],
                "random_seed": config["main"]["random_seed"],
                "val_size": config["data"]["test_size"],
                "stratify_by": config["data"]["stratify_by"],
                "max_tfidf_features": config["random_forest_pipeline"]["max_tfidf_features"]})
            

    if "test_regression_model" in steps_to_execute:
        _ = mlflow.run(
            os.path.join(root_path, "test_regression_model"),
            entry_point="main",
            parameters={
                "mlflow_model": f"{config['random_forest_pipeline']['output_artifact']}:prod", 
                "test_dataset": config["data"]["test_data"]})# "data_test.csv:latest"})


if __name__ == "__main__":
    go()
