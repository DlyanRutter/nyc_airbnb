"""
1)
main.py in main dir
########Why couldn't we do this?"
Gives error: 023/01/31 09:39:05 INFO mlflow.projects.backend.local: === Running command 'source /Users/dylanrutter/opt/anaconda3/bin/../etc/profile.d/conda.sh && 
conda activate mlflow-f2cd21b0ccf4eab994ac56c56aa7b85527d4a6f3 1>&2 && python main.py main.steps=\'all\' $(echo 'main.execute_steps='"'"'download'"'"'')' 
in run with ID 'fc2161f4149b4436bf0d2b55768b6378' === 
Could not override 'main.steps'.
To append to your config use +main.steps='all'
Key 'steps' is not in struct
    full_key: main.steps
    object_type=dict

Config.yaml had been changed to
main:
  components_repository: "https://github.com/udacity/build-ml-pipeline-for-short-term-rental-prices#components"
  src_repository: "https://github.com/udacity/build-ml-pipeline-for-short-term-rental-prices#src" #mine
  # All the intermediate files will be copied to this directory at the end of the run.
  # Set this to null if you are running in prod
  project_name: nyc_airbnb
  experiment_name: development
  #steps: all
  execute_steps:
    - download
    - basic_cleaning
    - check_data
    - data_split
    - train_random_forest
"""
# still need metadata and pytest
import mlflow, os, hydra, json
from omegaconf import DictConfig, OmegaConf
@hydra.main(config_name='config', config_path='config', version_base='2.5')
def go(config: DictConfig):

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # Steps to execute
    #steps_par = config['main']['steps']
    #active_steps = steps_par.split(",") if steps_par != "all" else _steps
    root_path = hydra.utils.get_original_cwd()
    if isinstance(config["main"]["execute_steps"], str):
        steps_to_execute = config["main"]["execute_steps"].split(",")
    else:
        #print (type(config["main"]["execute_steps"]))
        #assedrt isinstance(config["main"]["execute_steps"], list)
        steps_to_execute = config["main"]["execute_steps"]


    if "download" in steps_to_execute:#active_steps:
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
            ##################
            # Implement here #
            ##################
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
            ##################
            # Implement here #
            ##################
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

            ##################
            # Implement here #
            ##################
            

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
            entry_point="main", #might have an issue here
            parameters={
                "trainval_artifact": "data_train.csv:latest",
                "rf_config": rf_config,
                "output_artifact": config["random_forest_pipeline"]["output_artifact"],
                "random_seed": config["main"]["random_seed"],
                "val_size": config["data"]["test_size"],
                "stratify_by": config["data"]["stratify_by"],
                "max_tfidf_features": config["random_forest_pipeline"]["max_tfidf_features"]})
            
  # DO NOT TOUCH

            # NOTE: use the rf_config we just created as the rf_config parameter for the train_random_forest
            # step

            ##################
            # Implement here #
            ##################

  

    if "test_regression_model" in steps_to_execute:
        _ = mlflow.run(
            os.path.join(root_path, "test_regression_model"),
            entry_point="main",
            parameters={
                "mlflow_model": f"{config['random_forest_pipeline']['output_artifact']}:prod", #was latest
                "test_dataset": "data_test.csv:latest"})


            ##################
            # Implement here #
            ##################

if __name__ == "__main__":
    go()
