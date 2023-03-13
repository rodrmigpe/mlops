#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from azureml.core.run import Run
from azureml.core import Dataset, Datastore, Workspace
from azureml.data import DataType
import os
import argparse
import joblib
import pickle
import json
from train import split_data, train_model, get_model_metrics


# In[ ]:


def register_dataset(
    aml_workspace: Workspace,
    dataset_name: str,
    datastore_name: str,
    file_path: str
) -> Dataset:
    datastore = Datastore.get(aml_workspace, datastore_name)

    # use `set_column_types` to set column data types
    data_types = {
       'municipality_numeric_identifier': DataType.to_long(),
       'unemp': DataType.to_float(),
       'age_mean': DataType.to_float(),
       'age_std': DataType.to_float(),
       'age_pct05': DataType.to_float(),
       'age_pct25': DataType.to_float(),
       'age_pct50': DataType.to_float(),
       'age_pct75': DataType.to_float(),
       'age_pct95': DataType.to_float(),
       'child_mean': DataType.to_float(),
       'frac_women': DataType.to_float(),
       'car1_custom_frac': DataType.to_float(),
       'car1_MKL_frac': DataType.to_float(),
       'car1_OMK_frac': DataType.to_float(),
       'car1_KWA_frac': DataType.to_float(),
       'car1_CAB_frac': DataType.to_float(),
       'car1_VAN_frac': DataType.to_float(),
       'car1_UMK_frac': DataType.to_float(),
       'car1_SUV_frac': DataType.to_float(),
       'car1_CPE_frac': DataType.to_float(),
       'car1_MIC_frac': DataType.to_float(),
       'car1_LKL_frac': DataType.to_float(),
       'car1_price_mean': DataType.to_float(),
       'car1_price_std': DataType.to_float(),
       'car1_price_pct05': DataType.to_float(),
       'car1_price_pct25': DataType.to_float(),
       'car1_price_pct50': DataType.to_float(),
       'car1_price_pct75': DataType.to_float(),
       'car1_price_pct95': DataType.to_float(),
       'car1_matriculation_pct05': DataType.to_float(),
       'car1_matriculation_pct25': DataType.to_float(),
       'car1_matriculation_pct50': DataType.to_float(),
       'car1_matriculation_pct75': DataType.to_float(),
       'car1_matriculation_pct95': DataType.to_float(),
       'car1_ccm_mean': DataType.to_float(),
       'car1_ccm_std': DataType.to_float(),
       'car1_ccm_pct05': DataType.to_float(),
       'car1_ccm_pct25': DataType.to_float(),
       'car1_ccm_pct50': DataType.to_float(),
       'car1_ccm_pct75': DataType.to_float(),
       'car1_ccm_pct95': DataType.to_float(),
       'car1_claim_mean': DataType.to_float(),
       'car1_claim_std': DataType.to_float(),
       'car1_claim_pct05': DataType.to_float(),
       'car1_claim_pct25': DataType.to_float(),
       'car1_claim_pct50': DataType.to_float(),
       'car1_claim_pct75': DataType.to_float(),
       'car1_claim_pct95': DataType.to_float(),
       'car1_sumcl_mean': DataType.to_float(),
       'car1_sumcl_std': DataType.to_float(),
       'car1_sumcl_pct05': DataType.to_float(),
       'car1_sumcl_pct25': DataType.to_float(),
       'car1_sumcl_pct50': DataType.to_float(),
       'car1_sumcl_pct75': DataType.to_float(),
       'car1_sumcl_pct95': DataType.to_float(),
       'car1_prem_mean': DataType.to_float(),
       'car1_prem_std': DataType.to_float(),
       'car1_prem_pct05': DataType.to_float(),
       'car1_prem_pct25': DataType.to_float(),
       'car1_prem_pct50': DataType.to_float(),
       'car1_prem_pct75': DataType.to_float(),
       'car1_prem_pct95': DataType.to_float(),
       'car1_age': DataType.to_float(),
       'age_0_19_census': DataType.to_float(),
       'age_20_64_census': DataType.to_float(),
       'age_65+_census': DataType.to_float(),
       'pop_census': DataType.to_long(),
       'n_customers': DataType.to_long()
   }

    dataset = Dataset.Tabular.from_delimited_files(path=(datastore, file_path), set_column_types=data_types)
    
    '''
    # Set the column types for the integer columns
    int_cols = ['0', '10', '63'] # assuming the first column is named '0', 11th is '10', and 64th is '63'
    dataset = dataset.set_column_types({col_name: 'int' for col_name in int_cols})

    # Set the column types for the float columns
    float_cols = [str(i) for i in range(68) if str(i) not in int_cols]
    dataset = dataset.set_column_types({col_name: 'float' for col_name in float_cols}) '''

    dataset = dataset.register(workspace=aml_workspace,
                               name=dataset_name,
                               create_new_version=True)

    return dataset


def main():
    print("Running train_aml.py")

    parser = argparse.ArgumentParser("train")
    parser.add_argument(
        "--model_name",
        type=str,
        help="Name of the Model",
        default="insurance_model.pkl",
    )

    parser.add_argument(
        "--data_file_path",
        type=str,
        help=("data file path, if specified,a new version of the dataset will be registered"),
        default="insurance",
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Dataset name",
        default="insurance_dataset",
    )

    args = parser.parse_args()

    print("Argument [model_name]: %s" % args.model_name)
    print("Argument [data_file_path]: %s" % args.data_file_path)
    print("Argument [dataset_name]: %s" % args.dataset_name)

    model_name = args.model_name
    data_file_path = args.data_file_path
    dataset_name = args.dataset_name

    run = Run.get_context()
    
    # Get the dataset
    if (dataset_name):
        if (data_file_path == 'none'):
            dataset = Dataset.get_by_name(run.experiment.workspace, dataset_name)  # NOQA: E402, E501
        else:
            dataset = register_dataset(run.experiment.workspace,
                                       dataset_name,
                                       "workspaceblobstore",
                                       data_file_path)
    else:
        e = ("No dataset provided")
        print(e)
        raise Exception(e)

    # Link dataset to the step run so it is trackable in the UI
    run.input_datasets['training_data'] = dataset
    #run.parent.tag("dataset_id", value=dataset.id)

    # Split the data into test/train
    df = dataset.to_pandas_dataframe()
    
    # Test if dataset has Nulls, Duplicates or Negatives
    #test_null_values(df)
    #check_duplicates(df)
    #test_positive_values(df)
    
    #Test split_data function
    #test_split_data(df)
    #test_train_data_length(df)
    #test_valid_data_length(df)
    #test_correct_data_types(df)
    
    data = split_data(df)
    
    #Test train_model function
    #test_rmse_type(data)
    #test_model_type(data)
    #test_train_model(data)
    #test_max_rmse(data)

    # Train the model
    model = train_model(data)
    
    # Evaluate and log the metrics returned from the train function
    metrics = get_model_metrics(model, data)
    for (k, v) in metrics.items():
        run.log(k, v)
        #run.parent.log(k, v)

    # Also upload model file to run outputs for history
    os.makedirs('outputs', exist_ok=True)
    output_path = os.path.join('outputs', model_name)
    joblib.dump(value=model[0], filename=output_path)

    run.tag("run_type", value="train")
    print(f"tags now present for run: {run.tags}")

    # upload the model file explicitly into artifacts
    print("Uploading the model into run artifacts...")
    run.upload_file(name="./outputs/models/" + model_name, path_or_stream=output_path)
    print("Uploaded the model {} to experiment {}".format(model_name, run.experiment.name))
    dirpath = os.getcwd()
    print(dirpath)
    print("Following files are uploaded ")
    print(run.get_file_names())

    run.complete()


if __name__ == '__main__':
    main()

