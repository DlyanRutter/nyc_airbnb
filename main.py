# main pipeline file

import mlflow, os, hydra, json
from omegaconf import DictConfig, OmegaConf
@hydra.main(config_name='config', config_path='config', version_base='2.5')
def go(config: DictConfig):

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # Steps to execute
    root_path = hydra.utils.get_original_cwd()
    if isinstance(config["main"]["execute_steps"], str):
        steps_to_execute = config["main"]["execute_steps"].split(",")
    else:
        steps_to_execute = config["main"]["execute_steps"]


    if "download" in steps_to_execute:
        # Download file and load in W&B
        _ = mlflow.run(
            #os.path.join(root_path, "download"),
            f"{config['main']['components_repository']}/get_data",
            version="main",
            entry_point="main",
            parameters={
                    "sample": config["data"]["sample"], #"file_url": config["data"]#["file_url"], #taken from hydra
                    "artifact_name": "sample.csv",
                    "artifact_type": "raw_data",
                    "artifact_description": "Raw file as downloaded"
                },
            )

    if "preprocess" in steps_to_execute:
        # minor preprocessing
        _ = mlflow.run(
            os.path.join(root_path, "preprocess"),
            entry_point="main",
            parameters={
                "input_artifact": "sample.csv:latest",
                "artifact_name": "preprocessed_data.csv",
                "artifact_type": "preprocessed_data",
                "artifact_description": "Data with preprocessing applied",
                "min_price": config["data"]["min_price"],
                "max_price": config["data"]["max_price"]})

    if "check_data" in steps_to_execute:
        # tests to confirm data is valid
        _ = mlflow.run(
            os.path.join(root_path, "check_data"),
            entry_point="main",
            parameters={
                "reference_artifact": config["data"]["reference_dataset"],
                "sample_artifact": "preprocessed_data.csv:latest",
                "kl_threshold": config["data"]["kl_threshold"],
                "min_price": config["data"]["min_price"],
                "max_price": config["data"]["max_price"]})

    if "segregate" in steps_to_execute:
        # split into train and test sets
        _ = mlflow.run(
            os.path.join(root_path, "segregate"),
            entry_point="main",
            parameters={
                "input_artifact": "preprocessed_data.csv:latest",
                "artifact_root": "data",
                "artifact_type": "segregated_data",
                "test_size": config["data"]["test_size"],
                "stratify": config["data"]["stratify"]
            })

    if "random_forest" in steps_to_execute:
        # train model
        # NOTE: we need to serialize the random forest configuration into JSON
        model_config=os.path.abspath("random_forest_config.yml")
        rf_config = os.path.abspath("rf_config.json")
        with open(rf_config, "w+") as fp:
            json.dump(dict(config["random_forest_pipeline"]["random_forest"].items()), fp)
        with open(model_config, "w+", encoding="utf-8") as fp:
            fp.write(OmegaConf.to_yaml(config["random_forest_pipeline"]))
        _ = mlflow.run(
            os.path.join(root_path, "random_forest"),
            entry_point="main", #might have an issue here
            parameters={
                "trainval_artifact": "data_train.csv:latest",
                "rf_config": rf_config,
                "output_artifact": config["random_forest_pipeline"]["output_artifact"],
                "random_seed": config["main"]["random_seed"],
                "val_size": config["data"]["test_size"],
                "stratify_by": config["data"]["stratify_by"],
                "max_tfidf_features": config["random_forest_pipeline"]["max_tfidf_features"]})
            
    if "test_regression_model" in steps_to_execute:
        # run model on test set
        _ = mlflow.run(
            os.path.join(root_path, "test_regression_model"),
            entry_point="main",
            parameters={
                "mlflow_model": f"{config['random_forest_pipeline']['output_artifact']}:prod", #was latest
                "test_dataset": "data_test.csv:latest"})


if __name__ == "__main__":
    go()
