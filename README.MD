# Risk Evaluation for Retail Banks

## Introduction
After researching, our startup team decided to utilize machine learning to provide risk evaluation as a service for retail banks. We have access to a dataset from Home Credit Group. We are interested in using this data to perform analysis and build various models to aid in providing risk evaluation as a service for retail banks.

## Business Objective
We aim to equip banks with machine learning tools for more accurate risk assessments, minimize financial losses from defaults, and streamline loan approval processes. For a more detailed version of our objectives and business plan, please visit our `Loan_Risk` notebook.

## Installation
Install the required Python packages:
  ```
  pip install -r requirements.txt
  ```

## Notebook Structure

**Part 1: Data Understanding & Cleaning**:
In this part, we gain an overview of the various datasets we are working with, check for anomalies, and prepare our data to streamline our analysis and modeling.

**Part 2: Exploratory Data Analysis**:
Our analysis section focused on important features related to loan late payment/default. This gave us a foundational understanding and helped prepare us to build meaningful machine-learning models.

**Part 3: Machine Learning Models**:
In this part, we developed various models for predicting whether a client will be late on payments. We tested different models, engineered various features, performed hyperparameter tuning, and evaluated our models.

## Model Deployment
Our final section consists of deploying our model to google cloud. We deployed our best-performing model because that will be the best way to demonstrate our product. All the code for this section is in the `flask_app` folder.

## API Usage
Our API endpoint can be accessed using various tools such as Hoppscotch or `curl`. Please note our endpoint accepts POST requests only.

**URL**: https://risk-evaluation-kqfr2svtzq-uc.a.run.app  
**Endpoint**: `POST /predict_risk`

**Example With Our URL**: To test our API, we have set up a demo page with the JSON file already loaded; you simply follow this link, https://hopp.sh/r/8oAvRhDGGshi, and click `send.`  If you are prompted to choose an interceptor please click `proxy` in order for the request to work. This will return a json that looks like this:`{"Probability of Late Payment": 57.55}`

**Example With Curl:**
There is also the option to use `curl` and get predictions directly from your command line.
```bash
curl -X POST https://risk-evaluation-kqfr2svtzq-uc.a.run.app/predict_risk \
-H "Content-Type: application/json" \
-d '{
      "EXT_SOURCE_2":0.4125171304,
      "EXT_SOURCE_3":0.1538025886,
      "EXT_SOURCE_1":0.6499323249,
      "AMT_ANNUITY":30226.5,
      AGE_YEARS":52.79,
      ... rest of the features
    }'
