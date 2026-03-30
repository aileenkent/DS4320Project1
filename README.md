# DS 4320 Project 1: Diabetic Patients Readmission Prediction 

Aileen Kent, sbx3sw

DOI:

Press Release [Link](https://github.com/aileenkent/DS4320Project1/blob/c346525a6540a8d698c99d23c0a0faa6eb43f3c3/Press%20Release.md)

Data folder [Link](https://myuva-my.sharepoint.com/:f:/g/personal/sbx3sw_virginia_edu/IgBww9hBIwRRRLA1D5kjBQJIAZgjIwPbDUQb0bT_flmnXrU?e=jLDLmF)

Pipeline Link 

License: GNU General Public License v3.0 [link](https://github.com/aileenkent/DS4320Project1/tree/main?tab=GPL-3.0-1-ov-file#)

## Problem Definition

Initial General Problem: How can healthcare systems use patient data to reduce preventable hospital readmissions?

Specific Problem: Can we classify diabetic patients as likely or unlikely to be readmitted within 30 days of discharge, using clinical and demographic features from the UCI Diabetes 130-US Hosptials dataset?

Rationale:
This refinement focuses on narrowing down the scope of the hospital readmissions being addressed by my project. The general problem of hospital readmissions includes all possible conditions, which makes it impratical for me to address. Focusing specifically on diabetic patients narrows the scope of the problem to a specific population that contains specific standardized records. The 30-day readmission window clarifies the definition of readmission and is the standard metric used by the Centers for Medicare & Medicaid Services. The UCI dataset contains over 100,000 patient encounters, making it a sufficient enough dataset for a classfication model.

Motivation: 
This project is motivated by the quality and cost problems that hospital readmissions within 30 days raise within the US healthcare system. Hospitals are penalized financially for excessive readmission rates by the Hospital Readmission Reduction Program. But it is crucial that patients aren't dismissed or have their symptoms reduced in order to avoide readmission. Diabetic patiens are among the highest-risk groups for readmission because the managment of diabetes is easily disrupted post-discharge. A machine learning classifier that can flag high-risk patients at discharge could trigger targeted interventions that could reduce readmission rates while imporivng patient outcomes.

Press Release Headline: Intervention Before You Leave - New Readmission Risk tool Protects Patients [Link](https://github.com/aileenkent/DS4320Project1/blob/c346525a6540a8d698c99d23c0a0faa6eb43f3c3/Press%20Release.md)

## Domain Exposition

Terminology:

| Term | Definition |
| ---- | ---- |
| 30-Day Readmission | A patient returning to the hospital within 30 days of discharge |
| HRRP | Hospital Readmission Reduction Program - CMS program penalizing hospitals for excess readmissions |
| HbA1c | Hemoglobin A1c - blood marker reflecting average blood sugar over ~ 3 months |
| Discharge Disposition | Where a patiente is sent after leaving the hospital |
| LOS | Length of Stay - number of days a patient was hopsitalized |
| Primary Diagnosis (ICD) | The main coded diagnosis driving hospitalization (ICD 9/10 system) |
| Polypharmacy | Concurrent use of multiple medications |
| Case Management | Hospital service coordinating post-discharge care plans |
| KPI: Recall (Sensitivity) | Proportion of true readmissions correctly identified by the model |
| KPI: Readmission Rate | Percentage of patients readmitted within 30 days |
| SNF | Skilled Nursing Facility - post-acute care destination |
| CMS | Centers for Medicare & Medicaid Services - US federal health insurer/regulator |

Domain:

This project is in the domain of clinical healthcare analytics, specifically hospital quality improvement and care transitions management. Hospitals in the US are reimbursed and evaluated by CMS partially based on their readmission rates. Diabetes is one of the most common and costly chronic conidtions in the US, with diabetic patients showing disporportionately high readmission rates. With electronic health record (EHR) data becoming stanrdarized, machine learning models can be trained on historical records and integrated into clinical decision support system.

Background Reading Folder [Link](https://myuva-my.sharepoint.com/:f:/g/personal/sbx3sw_virginia_edu/IgAMrnGCX7x6SL94d1PQ4Tg4AXeTnslZlwXzU8QLjjM84jo?e=dSbtx6)

| Title | Description | Link |
| ---- | ---- | ---- |
| UCI ML Diabetes 130-US Hospitals Datset (Strack et al., 2014) | Original paper describing dataset used in project | [link](https://myuva-my.sharepoint.com/:b:/r/personal/sbx3sw_virginia_edu/Documents/DS4320Project1Reading/BioMed%20Research%20International%20-%202014%20-%20Strack%20-%20Impact%20of%20HbA1c%20Measurement%20on%20Hospital%20Readmission%20Rates%20%20Analysis%20of%2070.pdf?csf=1&web=1&e=7Gs3hP) |
| CMS Hospital Readmissions Reduction Program Overview | Explains the policy context and financial stakes of readmissions | [link](https://myuva-my.sharepoint.com/:u:/r/personal/sbx3sw_virginia_edu/Documents/DS4320Project1Reading/CMS%20Hospital%20Readmission%20Reduction%20Program%20Overview.url?csf=1&web=1&e=EsNxn2) |
| A Systemic Review of Recent Studies on Hospital Readmissions of Patients With Diabetes (Kukde et al., 2024) | Reivews 21 studies from 2015-2023  examining risk factors, prediction methods, and outcomes for diabetic hospital readmissions | [link](https://myuva-my.sharepoint.com/:b:/r/personal/sbx3sw_virginia_edu/Documents/DS4320Project1Reading/A%20Systemic%20Review%20of%20Recent%20Studies%20on%20Hospital%20Readmissions%20of%20Patients%20With%20Diabetes.pdf?csf=1&web=1&e=HLxVsa) |
| ADA Standards of Medical Care in Diabetes | Clinical guidelines context for what "good" diabetes management looks like | [link](https://myuva-my.sharepoint.com/:b:/r/personal/sbx3sw_virginia_edu/Documents/DS4320Project1Reading/ADA%20Standards%20of%20Medical%20Care%20in%20Diabetes.pdf?csf=1&web=1&e=098e2z) |
| Predicting and Preventing Acute Care Re-Utilization by Patients with Diabetes (Rubin & Shah, 2021) | Provides a review, summary and evaulation of possible interventions | [link](https://myuva-my.sharepoint.com/:b:/r/personal/sbx3sw_virginia_edu/Documents/DS4320Project1Reading/Predicting%20and%20Preventing%20Acute%20Care%20Re-Utilization%20by%20Patients%20with%20Diabetes%20(Rubin%20%26%20Shah,%202021).pdf?csf=1&web=1&e=zgbRXB)

## Data Creation

The project dataset is the UCI Machine Learning Repository: Diabetes 130-US Hosptials for Years 1999-2008. This dataset is publically avaliable online at https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008. The data was directly downloaded from the repository creating a raw file called `diabetic_data.csv` and a companion mapping file called `IDs_mapping.csv`. 

I then split the `IDs_mapping.csv` into three separate csv files for each type of id. These files are called `admission_type_id`, `discharge_disposition_id`, and `admission_source_id`. This decision was made because originally the three types of ids were stacked ontop of each other just visiually separated by a blank row (or a row of nan values), which made it very difficult to search the file without actually opening it and visually looking as the ids all use numerical codes starting at 1. 

Following this I split the `diabetic_data.csv` into two files, `diabetic_data.csv` and `medicines.csv`. The file `medicines.csv` would include the 23 columns of yes/no responses to individual medications in the original `diabetic_data.csv` as well as a copy of the encounter_id column to ensure I can match the medications to the individual encounter. This decision was made preserve the medication list while not having it be in the main table as I do not anticipate needing to use all of them. 

| file | Description | Link |
| ---- | ---- | ---- |
| `data_cleaning.py` | Loads raw `diabetic_data.csv`, splits the file into `diabetic_data.csv` and `medications.csv`, drops columns with >50% missing values (`weight`, `payer_code`, `medical_specialty`), indicator feature `weight_missing` added, replaces `?` with NaN, binarizes the target variable (`readmitted_30d`), encodes categorical features, and saves `diabetic_readmission_cleaned.csv` | [Link](https://github.com/aileenkent/DS4320Project1/blob/fb7d3e68343bacbfc136be439309dc8daf9b598f/Data%20Creation/data_cleaning.py) |
| `id_cleaning.py` | splits the original `IDs_mapping.csv` into three separate csv files, using the blank/NaN rows in the `IDs_mapping.csv` as the divider between the different types of IDs | [Link to repo](https://github.com/aileenkent/DS4320Project1/blob/fb7d3e68343bacbfc136be439309dc8daf9b598f/Data%20Creation/id_cleaning.py) |

Bias Identification:

There are a few sources of bias in this dataset. Firstly, thre is a demographic representation bias as the dataset is drawn from 130 US hospitals, so it doesn't represent rural hospital, critical access hospital, or non-US health systems. Also it the distribution of demographic data is a reflection of the hospital mix of 1999-2008. Secoundly, there is a survivorship bias in the `readmitted` label, as it only captures readmissions in the same hospital and does not account for patients who died shortly after discharge. Thirdly, there is three features with high missingness, icnluding `weight`, which is likely correlated with the outcomes as it is related to metabolic concerns. Fourthly, there is a historical bias present because this dataset is from 1999-2008, many practices and policies of medicine have changed since then.

Bias Mitigation:

To attempt to address these potential sources of bias, I will employ the following mitigation strategies. Firstly, the demographic representation bias will be acknowledged and results will be reported stratified by race and age group, not generalizable beyond the original hosptial populaiton. Secondly, survivorship bias in the `readmitted` label cannot be altered so it will be contintually documented as a limitation. Thirdly, the missing data in weight will be mitigated by treating `weight` missingness as an indicator feature `weight_missing`. Fourthly, the historial biases will be continually addressed in the discussion section and any deployment of the model would require retraining on contemporary data.

Rationale: 

There are a few cirtical decisions that were made. Firstly, I binarized the target variable to fit wiht the project problem statment that is a binary classfiication task. Secondly. I dropped the high-missingness columns as a decision to avoid introducing more noise than signal.

There is uncertainly introduced by the feature engineering as the ICD-9 grouping schema might cause different results as grouping schemas have that possibility.

## Metadata

![Entity Relationship Diagram with `diabetic_readmission_cleaned.csv`, `medications.csv`, `admission_type_id`, `discharge_disposition_id`, and `admission_source_id`](Images/Project1ERD.png)

| Table Name | Description | Link to CSV |
|---|---|---|
| `diabetic_readmission_cleaned.csv` | Cleaned, first-encounter-only subset of the UCI Diabetes 130-US Hospitals dataset. One row per unique patient (~71,518 records). Binary target `readmitted_30d` indicates 30-day readmission. | [Link](https://myuva-my.sharepoint.com/:x:/r/personal/sbx3sw_virginia_edu/Documents/DS4030Project1/diabetic_readmission_cleaned.csv?d=w04a956f286b340ed8006089ecf825c77&csf=1&web=1&e=nMSq3W) |
| `medicines.csv` | one row per unique patient, only responses to questions about individual medications, encounter_id is FK to connect to `diabetic_readmisison_cleaned.csv` | [Link](https://myuva-my.sharepoint.com/:x:/r/personal/sbx3sw_virginia_edu/Documents/DS4030Project1/medicines.csv?d=w2e2c742ad7fa4a7d943cac8f288dac8a&csf=1&web=1&e=5wLm6V) |
| `admission_source_id.csv` | description of the code for admission source ids, includes the numeric number that is a foreign key in the `diabetic_readmission_cleaned.csv` and description | [Link](https://myuva-my.sharepoint.com/:x:/r/personal/sbx3sw_virginia_edu/Documents/DS4030Project1/admission_source_id.csv?d=w4b0dc0b7158b406584fa2047fd843699&csf=1&web=1&e=hAPylx) |
| `admission_type_id.csv` | description of the code for admission type ids, includes the number number that is a foreign key in the `diabetic_readmission_cleaned.csv` and description | [Link](https://myuva-my.sharepoint.com/:x:/r/personal/sbx3sw_virginia_edu/Documents/DS4030Project1/admission_type_id.csv?d=w78cfb81f3b934129b4a8e7185ac427f7&csf=1&web=1&e=jBNmGP) |
| `discharge_disposition_id.csv` | discription of the code for discharge disposition ids, includes the number number that is a foreign key in the `diabetic_readmission_cleaned.csv` and description | [Link](https://myuva-my.sharepoint.com/:x:/r/personal/sbx3sw_virginia_edu/Documents/DS4030Project1/discharge_disposition_id.csv?d=w70f6676d9f5e4da086393ba6f0550ccb&csf=1&web=1&e=2sXBEd) |

| Table | Feature Name | Data Type | Description | Example |
| ---- |---|---|---|---|
| `diabetic_readmission_cleaned.csv` | encounter_id | Integer | PK - Unique identifier for each hospital encounter | `2278392` |
| `diabetic_readmission_cleaned.csv` | patient_nbr | Integer | De-identified patient identifier; used to filter to first encounter per patient | `8222157` |
| `diabetic_readmission_cleaned.csv` | race | String | Patient's self-reported race category | `"Caucasian"` |
| `diabetic_readmission_cleaned.csv` | gender | String | Patient's recorded gender | `"Female"` |
| `diabetic_readmission_cleaned.csv` | age | String | Age bracket in 10-year intervals | `"[50-60)"` |
| `diabetic_readmission_cleaned.csv` | admission_type_id | Integer | FK - Coded admission type (1=Emergency, 2=Urgent, 3=Elective, etc.) | `1` |
| `diabetic_readmission_cleaned.csv` | discharge_disposition_id | Integer | FK - Coded discharge destination (1=Home, 3=SNF, 11=Deceased, etc.) | `1` |
| `diabetic_readmission_cleaned.csv` | admission_source_id | Integer | FK - Coded source of admission (7=ER, 1=Physician Referral, etc.) | `7` |
| `diabetic_readmission_cleaned.csv` | time_in_hospital | Integer | Number of days patient remained hospitalized | `5` |
| `diabetic_readmission_cleaned.csv` | num_lab_procedures | Integer | Number of distinct lab procedures performed during encounter | `41` |
| `diabetic_readmission_cleaned.csv` | num_procedures | Integer | Number of non-lab procedures performed during encounter | `1` |
| `diabetic_readmission_cleaned.csv` | num_medications | Integer | Number of distinct medications administered during encounter | `18` |
| `diabetic_readmission_cleaned.csv` | number_outpatient | Integer | Number of outpatient visits in the year prior to the encounter | `0` |
| `diabetic_readmission_cleaned.csv` | number_emergency | Integer | Number of emergency department visits in the year prior to the encounter | `0` |
| `diabetic_readmission_cleaned.csv` | number_inpatient | Integer | Number of inpatient visits in the year prior to the encounter | `0` |
| `diabetic_readmission_cleaned.csv` | number_diagnoses | Integer | Number of distinct diagnoses entered for the encounter | `9` |
| `diabetic_readmission_cleaned.csv` | diag_1_group | String | ICD-9 category group for primary diagnosis | `"Circulatory"` |
| `diabetic_readmission_cleaned.csv` | diag_2_group | String | ICD-9 category group for secondary diagnosis | `"Diabetes"` |
| `diabetic_readmission_cleaned.csv` | diag_3_group | String | ICD-9 category group for additional diagnosis | `"Musculoskeletal"` |
| `diabetic_readmission_cleaned.csv` | A1Cresult | String | HbA1c test result: `None`, `Norm`, `>7`, or `>8` | `">8"` |
| `diabetic_readmission_cleaned.csv` | insulin | String | Insulin medication dosage change status: `No`, `Steady`, `Up`, `Down` | `"Steady"` |
| `diabetic_readmission_cleaned.csv` | change | String | Whether any medication dosage was changed: `Ch` or `No` | `"Ch"` |
| `diabetic_readmission_cleaned.csv` | diabetesMed | String | Whether any diabetes medication was prescribed: `Yes` or `No` | `"Yes"` |
| `diabetic_readmission_cleaned.csv` | weight_missing | Integer | Indicator variable: 1 if weight was not recorded, 0 if recorded | `1` |
| `diabetic_readmission_cleaned.csv` | readmitted_30d | Integer | **Target variable.** 1 = patient readmitted within 30 days of discharge; 0 = not readmitted within 30 days | `0` |
| medicines.csv` | encounter_id | Integer | FK - Unique identifier for each hospital encounter | `2278392` |
| medicines.csv` | metformin | String | Indicator if patiente takes metformin (either no or steady) | `steady` |
| medicines.csv` | repaglinide | String | Indicator if patiente takes repaglinide (either no or steady) | `no` |
| medicines.csv` | nateglinide | String | Indicator if patiente takes nateglinide (either no or steady) | `steady` |
| medicines.csv` | chlorpropamide | String | Indicator if patiente takes chlorpropamide (either no or steady) | `no` |
| medicines.csv` | glimepiride | String | Indicator if patiente takes glimepiride (either no or steady) | `steady` |
| medicines.csv` | acetohexamide | String | Indicator if patiente takes acetohexamide (either no or steady) | `steady` |
| medicines.csv` | glipizide | String | Indicator if patiente takes glipizide (either no or steady) | `no` |
| medicines.csv` | glyburide | String | Indicator if patiente takes glyburide (either no or steady) | `no` |
| medicines.csv` | tolbutamide | String | Indicator if patiente takes tolbutamide (either no or steady) | `no` |
| medicines.csv` | piolitazone | String | Indicator if patiente takes piolitazone (either no or steady) | `steady` |
| medicines.csv` | rosiglitazone | String | Indicator if patiente takes rosiglitazone (either no or steady) | `steady` |
| medicines.csv` | acarbose | String | Indicator if patiente takes acarbose (either no or steady) | `no` |
| medicines.csv` | miglitol | String | Indicator if patiente takes miglitol (either no or steady) | `steady` |
| medicines.csv` | troglitazone | String | Indicator if patiente takes troglitazone (either no or steady) | `no` |
| medicines.csv` | tolazamide | String | Indicator if patiente takes tolazamide (either no or steady) | `no` |
| medicines.csv` | examide | String | Indicator if patiente takes examide (either no or steady) | `steady` |
| medicines.csv` | citoglipton | String | Indicator if patiente takes citoglipton (either no or steady) | `steady` |
| medicines.csv` | insulin | String | Indicator of the patient's use of insulin (no, up, steady, down) | `up` |
| medicines.csv` | glyburide-metformin | String | Indicator if patiente takes glyburide-metformin (either no or steady) | `steady` |
| medicines.csv` | glipizide-metformin | String | Indicator if patiente takes glipizide-metformin (either no or steady) | `no` |
| medicines.csv` | glimepride-pioglitazone | String | Indicator if patiente takes glimepride-pioglitazone (either no or steady) | `steady` |
| medicines.csv` | metformin-rosiglitazone | String | Indicator if patiente takes metformin-rosiglitazone (either no or steady) | `steady` |
| medicines.csv` | metformin-pioglitazone | String | Indicator if patiente takes metformin-pioglitazone (either no or steady) | `no` |
| `admission_type_id.csv` | admission_type_id | Integer | PK - Coded numerical admission type indicator | `1` |
| `admission_type_id.csv` | description | String | Explanation of each coded number | `Elective` |
| `discharge_disposition_id.csv` | discharge_disposition_id | Integer | PK - Coded numerical discharge disposition type indicator | `4` |
| `discharge_disposition_id.csv` | description | String | Explanation of each coded number | `Discharged/transferred to ICF` |
| `admission_source_id.csv` | admission_source_id | Integer | Coded numerical admission source type indicator | `10` |
| `admission_source_id.csv` | description | String | Explanation of each coded number | `Transfer from critial access hospital` |

| Table | Feature | Min | Max | Mean | Std Dev | Notes |
| ---- |---|---|---|---|---|---- |
| `diabetic_readmission_cleaned.csv` | time_in_hospital | 1 | 14 | 4.4 | 2.99 | Capped at 14 in dataset; any stays >14 days recorded as 14 |
| `diabetic_readmission_cleaned.csv` | num_lab_procedures | 1 | 132 | 43.1 | 19.67 | Right-skewed; high values may reflect prolonged stays rather than severity |
| `diabetic_readmission_cleaned.csv` | num_procedures | 0 | 6 | 1.34 | 1.71 | Zero-inflated; many patients have no non-lab procedures |
| `diabetic_readmission_cleaned.csv` | num_medications | 1 | 81 | 16.0 | 8.13 | High values are meaningful signal but may also reflect data entry patterns |
| `diabetic_readmission_cleaned.csv` | number_outpatient | 0 | 42 | 0.37 | 1.34 | Heavily right-skewed; most patients have 0 |
| `diabetic_readmission_cleaned.csv` | number_emergency | 0 | 76 | 0.20 | 0.93 | Extreme outliers (>10) may reflect data errors |
| `diabetic_readmission_cleaned.csv` | number_inpatient | 0 | 21 | 0.64 | 1.24 | Strong predictor of readmission but affected by missingness from cross-hospital transfers |
| `diabetic_readmission_cleaned.csv` | number_diagnoses | 1 | 16 | 7.42 | 1.94 | Near-normally distributed; less extreme than other count features |
