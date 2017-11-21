# Deep Learning for Predictive Maintenance real-word scenario.

Deep learning is one of the most popular trends in the machine learning space nowadays, and there are many fields and applications where it stands out, such as driverless cars, speech and image recognition, robotics and finance. Deep learning is a set of algorithms that is inspired by the shape of our brain (biological neural networks), and machine learning and cognitive scientists usually refer to it as Artificial Neural Networks (ANN).

Predictive maintenance is also a very popular area where many different techniques are designed to help determine the condition of an equipment in order to predict when maintenance should be performed. Predictive maintenance encompasses a variety of topics, including but not limited to: failure prediction, failure diagnosis (root cause analysis), failure detection, failure type classification, and recommendation of mitigation or maintenance actions after failure.

In predictive maintenance scenarios, data is collected over time to monitor the state of an equipment with the final goal of finding patterns to predict failures. Among the deep learning methods, [Long Short Term Memory (LSTM)](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) networks are especially appealing to the predictive maintenance domain due to the fact that they are very good at learning from sequences. This fact lends itself to their applications using time series data by making it possible to look back for longer periods of time to detect failure patterns.

In this tutorial, we build an LSTM network for the data set and scenerio described at [Predictive Maintenance](https://gallery.cortanaintelligence.com/Collection/Predictive-Maintenance-Template-3) to predict remaining useful life of aircraft engines. In summary, the template uses simulated aircraft sensor values to predict when an aircraft engine will fail in the future so that maintenance can be planned in advance.

This tutorial uses [keras](https://keras.io/) deep learning library with Microsoft Cognitive Toolkit [CNTK](https://docs.microsoft.com/en-us/cognitive-toolkit/Using-CNTK-with-Keras)as backend.

## Link to the Gallery GitHub repository 
Following is the link to the public GitHub repository:
[https://github.com/Azure/MachineLearningSamples-DeepLearningforPredictiveMaintenance](https://github.com/Azure/MachineLearningSamples-DeepLearningforPredictiveMaintenance)

## Use case overview
This tutorial uses the example of simulated aircraft engine run-to-failure events to demonstrate the predictive maintenance modeling process. The implicit assumption of modeling data as done below is that the asset of interest has a progressing degradation pattern, which is reflected in the asset's sensor measurements. By examining the asset's sensor values over time, the machine learning algorithm can learn the relationship between the sensor values and changes in sensor values to the historical failures in order to predict failures in the future. We suggest examining the data format and going through all three steps of the template before replacing the data with your own.

## Prerequisites

- An [Azure account](https://azure.microsoft.com/free/) (free trials are available).
- An installed copy of Azure Machine Learning Workbench with a workspace created.
- For model operationalization: Azure Machine Learning Operationalization with a local deployment environment setup and a [model management account](https://docs.microsoft.com/en-us/azure/machine-learning/preview/model-management-overview)

## Create a new Workbench project
Create a new project using this example as a template:

1. Open Azure Machine Learning Workbench
2. On the Projects page, click the + sign and select New Project
3. In the Create New Project pane, fill in the information for your new project
4. In the Search Project Templates search box, type "Predictive Maintenance" and select the template
5. Click Create

## Prepare the notebook server computation target

To run on your local machine, from the AML Workbench `File` menu, select either the `Open Command Prompt` or `Open PowerShell` CLI. Within the CLI windows execute the following commands:

`az ml experiment prepare --target docker --run-configuration docker`

 We suggest running on a  DS4_V2 standard [Data Science Virtual Machine for Linux (Ubuntu)](https://azuremarketplace.microsoft.com/en-us/marketplace/apps/microsoft-ads.linux-data-science-vm-ubuntu). Once the DSVM is configured, you need to run the following two commands:

`az ml computetarget attach --name [Desired_Connection_Name] --address [VM_IP_Address] --username [VM_Username] --password [VM_UserPassword] --type remotedocker`

`az ml experiment prepare --target [Desired_Connection_Name] --run-configuration [Desired_Connection_Name]`

With the docker images _prepared_, open the jupyter notebook server either within the *AML Workbench* notebooks tab, or to start a browser-based server, run:
`az ml notebook start`

- Notebooks are stored in the `Code` directory found in the Jupyter environment. We run these notebooks sequentially as numbered, starting on (`Code\1_data_ingestion.ipynb`).

- Select the kernel to match your [Project_Name]_Template [Desired_Connection_Name] and click Set Kernel

## Data description 
The template takes three datasets as inputs. 
1. Train data: It is the aircraft engine run-to-failure data.
2. Test data: It is the aircraft engine operating data without failure events recorded.
3. Truth data: It contains the information of true remaining cycles for each engine in the testing data. 

The input data consists of "PM_train.txt", "PM_test.txt", and "PM_truth.txt" in the original data source. 
The training data ("PM_train.txt") consists of multiple multivariate time series with "cycle" as the time unit, together with 21 sensor readings for each cycle. Each time series can be assumed as being generated from a different engine of the same type. Each engine is assumed to start with different degrees of initial wear and manufacturing variation, and this information is unknown to the user. In this simulated data, the engine is assumed to be operating normally at the start of each time series. It starts to degrade at some point during the series of the operating cycles. The degradation progresses and grows in magnitude. When a predefined threshold is reached, then the engine is considered unsafe for further operation. In other words, the last cycle in each time series can be considered as the failure point of the corresponding engine. Taking the sample training data shown in the following table as an example, the engine with id=1 fails at cycle 192, and engine with id=2 fails at cycle 287. 
The testing data ("PM_test.txt") has the same data schema as the training data. The only difference is that the data does not indicate when the failure occurs (in other words, the last time period does NOT represent the failure point). Taking the sample testing data shown in the following table as an example, the engine with id=1 runs from cycle 1 through cycle 31. It is not shown how many more cycles this engine can last before it fails.
The ground truth data ("PM_truth.txt") provides the number of remaining working cycles for the engines in the testing data. Taking the sample ground truth data shown in the following table as an example, the engine with id=1 in the testing data can run another 112 cycles before it fails. 

### Data Labeling
Based on the input data description we have walked through in the previous section, an intuitive predictive maintenance question to ask is "Given these aircraft engine operation and failure events history, can we predict when an in-service engine will fail?"
For the purpose of this tutorial, we re-formulate this question into a simple question: "Is this engine going to fail within w1 cycles?

## Scenario structure

Task 1: Data Ingestion & Preparation

The Data Ingestion Jupyter Notebook in the `Code/1_data_ingestion_and_preparation.ipnyb` loads the three input data sets into pandas dataframe format, prepares the data for the modelling part and does some preliminary data visualization. The data is then transformed into `PySpark` format and stored in an Azure Blob storage container on your subscription for use in the next modelling task.

## Task 2: Model Building & Evaluation

The Model Building Jupyter Notebook in `Code/2_model_building_and_evaluation.ipnyb` that reads `PySpark` train and test data sets from blob storage. Then an LSTM network is built with the training data sets. The model performance is measured on the test set. The resulting model is serialized and stored in the local compute context for use in the operationalization task.

## Task 3: Operationalization

The operationalization Jupyter Notebook in `Code/3_operationalization.ipnyb` that takes the stored model and builds required functions and schema for calling the model on an Azure hosted web service. The notebook tests the functions, and zips the operationalization assets into a zip file that is also stored in your Azure Blob storage container. 

## Conclusion
This tutorial serves as a guide for beginners looking to apply deep learning in predictive maintenance domain within the Jupyter notebook environment in *Azure Machine Learning Workbench*. This tutorial uses a simple scenario where only one data source (sensor values) is used to make predictions. In more advanced predictive maintenance scenarios such as in [Predictive Maintenance Modelling Guide](https://gallery.cortanaintelligence.com/Notebook/Predictive-Maintenance-Modelling-Guide-R-Notebook-1), there are many other data sources (i.e. historical maintenance records, error logs, machine and operator features etc.) which may require different types of treatments to be used in the deep learning networks. Since predictive maintenance is not a typical domain for deep learning, its application is an open area of research.

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot automatically determines whether you need to provide a CLA and decorate the PR appropriately. You only need to follow the instructions provided by the bot across all Microsoft repository to use our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
More information is available at Code of Conduct FAQ or
contacts [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
