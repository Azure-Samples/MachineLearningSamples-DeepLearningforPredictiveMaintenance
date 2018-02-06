 # Deep Learning for Predictive Maintenance

The detailed documentation for this real world scenario includes the step-by-step walk-through:
https://docs.microsoft.com/en-us/azure/machine-learning/preview/scenario-deep-learning-for-predictive-maintenance

The public GitHub repository for this real world scenario contains all the code samples: https://github.com/Azure/MachineLearningSamples-DeepLearningforPredictiveMaintenance

## Introduction

Deep learning is one of the most popular trends in the machine learning space with applications to many eras including driverless cars, speech and image recognition, robotics and finance. Deep learning, also referred to as  Artificial Neural Networks (ANN), is a set of algorithms inspired by the shape of our brain (biological neural networks).

Predictive maintenance is uses machine learning methods to determine the condition of an equipment in order to preemptively trigger a maintenance visit to avoid adverse machine performance. In these scenarios, data is collected over time to monitor the state of an equipment with the final goal of finding patterns to predict failures. [Long Short Term Memory (LSTM)](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) networks are especially appealing to the predictive maintenance domain due to their ability at learning from sequences of data. LSTM are designed for application to time series data in order to look back over periods of time to detect temporal patterns that could lead to machine failures.

In this scenario, we build a LSTM network for the data set and scenario described at [Predictive Maintenance](https://gallery.cortanaintelligence.com/Collection/Predictive-Maintenance-Template-3) to predict remaining useful life of aircraft engines. In summary, the template uses simulated aircraft sensor values to predict when an aircraft engine will fail in the future so that maintenance can be planned in advance.

This tutorial uses [keras](https://keras.io/) deep learning library with [Tensorflow](https://www.tensorflow.org/) as the back end.

## Prerequisites

- An [Azure account](https://azure.microsoft.com/free/) (free trials are available).
- An installed copy of Azure Machine Learning Workbench with a work space created.
- Azure Machine Learning Operationalization with a local deployment environment setup and a [model management account](https://docs.microsoft.com/en-us/azure/machine-learning/preview/model-management-overview)

## Login

Once you install the AML Workbench app, we will need to connect to your Azure subscription. From the AML Workbench `File` menu, select either the `Open Command Prompt` or `Open PowerShell` command line interface (CLI). The CLI allows you to access your Azure services using the `az` commands. First, login to your Azure account with the command:

```
az login
``` 

This will generate a key to be used with the `https:\\aka.ms\devicelogin` URL. The CLI will block until the device login operation returns with the subscription summary.

## Let's Begin

To run on your local machine with [Docker](https://www.docker.com/) installed, from the AML Workbench `File` menu, select either the `Open Command Prompt` or `Open PowerShell` CLI. Within the CLI windows execute the following commands:

```az ml experiment prepare --target docker --run-configuration docker```

 We suggest running on a DS4_V2 standard [Data Science Virtual Machine for Linux (Ubuntu)](https://azuremarketplace.microsoft.com/en-us/marketplace/apps/microsoft-ads.linux-data-science-vm-ubuntu). Once the DSVM is configured, you need to run the following two commands:

```az ml computetarget attach remotedocker --name [Desired_Connection_Name] --address [VM_IP_Address] --username [VM_Username] --password [VM_UserPassword]```

```az ml experiment prepare --target [Desired_Connection_Name] --run-configuration [Desired_Connection_Name]```

With the docker images _prepared_, open the Jupyter notebook server either within the *AML Workbench* notebooks tab, or to start a browser-based server, run:
```az ml notebook start```

- The Jupyter Notebooks are stored in the `Code` directory seen in the notebook environment. We will run these notebooks sequentially as numbered, starting on (`Code\ 1_data_ingestion_and_preparation.ipynb `).

- Once the notebook server starts, select the kernel to match your [Project_Name]_Template [Desired_Connection_Name] and click Set Kernel

## Task 1: Data Ingestion & Preparation

The Data Ingestion Jupyter Notebook in the `Code/1_data_ingestion_and_preparation.ipnyb` loads the three input data sets into pandas dataframe format, prepares the data for the modelling and does some preliminary data visualization. The data sets are persisted to a local directory for use in the model building and evaluation task.

## Task 2: Model Building & Evaluation

The Model Building Jupyter Notebook in `Code/2_model_building_and_evaluation.ipnyb` reads the persisted training and test data sets from local storage and builds a LSTM network. The LSTM model is built using the training data set with two layers plus dropout to prevent overfitting. The model performance is measured on the test set. The resulting model is serialized and stored in the local compute context for use in the operationalization task.


## Task 3: Operationalization

The operationalization Jupyter Notebook in `Code/3_operationalization.ipnyb` takes the stored model and builds required functions and schema for calling the model on an Azure hosted web service. The notebook tests the functions,and zips the operationalization assets into a zip file which is copied to your Azure storage container in blob storage. Instructions for deploying the model assets and scoring some test data are included in the notebooks.

## Conclusion

This scenario serves as a guide to apply deep learning in predictive maintenance domain in *Azure Machine Learning Workbench*. This tutorial uses a simple scenario where only one data source (21 sensor values) is used to make these predictions. 

More advanced predictive maintenance scenarios are discussed in the [Predictive Maintenance Modelling Guide](https://gallery.cortanaintelligence.com/Notebook/Predictive-Maintenance-Modelling-Guide-R-Notebook-1). The modelling guide example includes multiple data sources (i.e. historical maintenance records, error logs, machine and operator features etc.) which may require different treatments to be used in with LSTM networks. 

# Data/Telemetry
 This advance scenario for *Deep Learning for Predictive Maintenance* collects usage data and sends it to Microsoft to help improve our products and services. Read our [privacy statement](https://privacy.microsoft.com/en-us/privacystatement) to learn more. 

# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot automatically determines whether you need to provide a CLA and decorate the PR appropriately. You only need to follow the instructions provided by the bot across all Microsoft repository to use our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
More information is available at Code of Conduct FAQ or
contacts [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
