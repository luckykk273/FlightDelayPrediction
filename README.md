# Flight Delay Prediction
## Introduction
A flight may delay for some unexpected accident.  
In these circumstances, a higher amount of claim should be assigned to the high-risk flights and on the other hand a lower amount of claim should be assigned to the low-risk flights.  
This is to adequately compensate the risk customers need to take.  
For example:  
If the delay time is greater than 3 hours OR is canceled, $800 will be claimed.  
Otherwise, claim amount will equal to $0.

## Objective
Use historical data with flight delay claims from 2013/01 to 2016/07 to predict claim amount for the future flights.

## Data Description
| Field Name | Description |
| :--------: | :---------- |
| flight_id	 | Unique ID for each flight |
| flight_no	 | flight number of each flight |
| Week	     | Indicate which week of year is the departure date in, For example, for flight departing at 17/1/2018, the week will be 3 |
| Departure	 | Location of departure |
| Arrival	 | Location of arrival |
| Std_hour	 | Scheduled departure time, in 24-hour format |
| delay_time | Number of delayed hours |
| is_claim	 | Claim amount, our insurance will pay customer a fixed amount of HK$800 when a delay happens. During your prediction, you can assign any value between $0 to $800 as the expected value of predicted claim amount. Absolute error and Brier error will then be calculated based on the difference between actual claim amount & your predicted claim amount. |

# Environment Setup
## Operating System
We run the whole project on Windows 10.

## Prerequisites
1. We use Python to do all the analysis work:  
Please ensure Python is installed before you follow the tutorial in this repository.  
If someone doesn't know how to install Python, please follow the guide in the official website: [Python Download](https://www.python.org/downloads/)
In this project, we use Python 3.7.6 to develop.(Highly recommend using Python 3.7.6 with us to prevent from version conflict.)

## Virtual Environment(Optional)
**NOTE: If someone doesn't want to build the virtual environment, go to step 4 and download this repository directly.**

1. We use **virtualenv** package to build a new virtual environment.  
Open cmd window and paste the command below:  
```pip3 install virtualenv```


2. Move to the directory where you want to build this project(e.g. C:\Users\User), 
paste the command below:  
```virtualenv flight_delay_prediction```  


3. Move into the path C:\Users\User\flight_delay_prediction in the console, and type the command below:  
```Scripts\activate```  
And the virtual environment will be activated successfully if there is a prefix at the beginning of the directory.  
( (flight_delay_prediction) C:\Users\User\flight_delay_prediction )


4. Download this repository to the directory you build the virtual environment before.

## Packages
Use command ```pip install -r requirements.txt``` to install all needed packages automatically for this project.  
All packages and corresponded version are listed below:  

| Packages | Version |
| :------: | :-----: |
| scikit-learn | 1.0.1 | 
| category-encoders | 2.3.0 |
| matplotlib | 3.3.0 |
| numpy | 1.21.4 |
| pandas | 1.1.0 |
| seaborn | 0.10.1 |
| Tensorflow | 1.14.0 |
| Keras | 2.2.5 |

# Project Structures
FlightDelayPrediction  
|----data  
| &nbsp;&nbsp;&nbsp; |----flight_delays_data.csv  
| &nbsp;&nbsp;&nbsp; |----train_features_normalized.csv  
| &nbsp;&nbsp;&nbsp; |----train_labels.csv  
| &nbsp;&nbsp;&nbsp; |----test_features_normalized.csv  
| &nbsp;&nbsp;&nbsp; |----test_labels.csv  
|  
|----example  
| &nbsp;&nbsp;&nbsp; |----logistic_regr.py  
| &nbsp;&nbsp;&nbsp; |----svm.py  
| &nbsp;&nbsp;&nbsp; |----ocsvm.py  
| &nbsp;&nbsp;&nbsp; |----mlp.py  
|  
|----model  
| &nbsp;&nbsp;&nbsp; |----logistic  
| &nbsp;&nbsp;&nbsp; |----svm  
| &nbsp;&nbsp;&nbsp; |----ocsvm  
| &nbsp;&nbsp;&nbsp; |----mlp_class_weights.h5  
|  
|----transform  
| &nbsp;&nbsp;&nbsp; |----target_Airline_encoder  
| &nbsp;&nbsp;&nbsp; |----target_Arrival_encoder  
| &nbsp;&nbsp;&nbsp; |----target_Departure_encoder  
| &nbsp;&nbsp;&nbsp; |----target_flight_no_encoder  
| &nbsp;&nbsp;&nbsp; |----z_score_scaler  
|  
|----utils  
| &nbsp;&nbsp;&nbsp; |----losses.py  
| &nbsp;&nbsp;&nbsp; |----metrics.py  
| &nbsp;&nbsp;&nbsp; |----utils.py  
|  
|----preprocessing.py  
|----README.md  
|----requirements.txt  
|----test.py  
|----utils.py  

# Examples
## Preprocessing Data
In the beginning, you only have the raw data **./data/flight_delays_data.csv**.
Execute the following command:  
```python data_exploration.py```  
Then you will get preprocessing data prepared:  
1. ./data/train_features_normalized.csv
2. ./data/train_labels.csv
3. ./data/test_features_normalized.csv
4. ./data/test_labels.csv.

## Model Training
In the example file, you can see four py file for different models training.  
Execute the following command(e.g. svm):  
```python svm.py```  
SVM model will start to be trained and will save model in **./model**.

## Predict Data
After models being trained, you can use test_ml() or test_dl() function in test.py file to predict other data.
You can easily pass two parameters: 
1. filepath: data you want to predict;
2. model_path: model you want to use to predict;

# Future Work
## How to process time-related variables  
There are lots of methods to process time-related variables.  
In our case, we transform time-related variables to periodic-time variables(e.g. day of year) because we assume flight delay is related to some trends in a year.  
Besides, we can also use year, month, day directly as variables.  
Cyclical encoding is another common way to capture patterns in a cycle.

## How to encode categorical variables
We use target encoding rather than one-hot encoding because one-hot encoding will increase the dimension of independent variables.  
But from the correlation matrix, we note that target encoding will enhance the relationship between independent variables and target variable.
Sometimes it is not a good phenomenon.
In fact, one-hot encoding is most used to encode categorical variables.  
Maybe we can consider to use one-hot encoding instead of target encoding. 

## Data imbalance
In the stage of data exploration, we can find that there are only 39363 flights are claimed(label 1) and 858037 flights are normal(label 0).
There is an extreme imbalance between two labels(about 1:20).
Though we use class weights when training SVM and MLP model(force labels to be balanced), the models still prefer to predict results to normal.
Resampling and one class SVM didn't resolve the problem of data imbalance, either.
It is a common problem when we face to data in the real world.
Some other methods can be tried in the future.