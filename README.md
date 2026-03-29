# DS 4320 Project 1: Diabetic Patients Readmission Prediction 

Aileen Kent
sbx3sw

DOI:

Press Release link

Data folder Link

Pipeline LInk 

License: GNU General Public License v3.0 [link](https://github.com/aileenkent/DS4320Project1/tree/main?tab=GPL-3.0-1-ov-file#)

## Problem Definition

Initial General Problem: How can healthcare systems use patient data to reduce preventable hospital readmissions?

Specific Problem: Can we classify diabetic patients as likely or unlikely to be readmitted within 30 days of discharge, using clinical and demographic features from the UCI Diabetes 130-US Hosptials dataset?

Rationale:
This refinement focuses on narrowing down the scope of the hospital readmissions being addressed by my project. The general problem of hospital readmissions includes all possible conditions, which makes it impratical for me to address. Focusing specifically on diabetic patients narrows the scope of the problem to a specific population that contains specific standardized records. The 30-day readmission window clarifies the definition of readmission and is the standard metric used by the Centers for Medicare & Medicaid Services. The UCI dataset contains over 100,000 patient encounters, making it a sufficient enough dataset for a classfication model.

Motivation: 
This project is motivated by the quality and cost problems that hospital readmissions within 30 days raise within the US healthcare system. Hospitals are penalized financially for excessive readmission rates by the Hospital Readmission Reduction Program. But it is crucial that patients aren't dismissed or have their symptoms reduced in order to avoide readmission. Diabetic patiens are among the highest-risk groups for readmission because the managment of diabetes is easily disrupted post-discharge. A machine learning classifier that can flag high-risk patients at discharge could trigger targeted interventions that could reduce readmission rates while imporivng patient outcomes.

Press Release Headline: Intervention Before You Leave - New Readmission Risk tool Protects Patients

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

Background Reading Folder Link

| Title | Description | Link |
| ---- | ---- | ---- |
| UCI ML Diabetes 130-US Hospitals Datset (Strack et al., 2014) | Original paper describing dataset used in project | [link](https://drive.google.com/file/d/1xd2hBvLsJuCnL7-ww2EuTq-vXk4DGs6_/view?usp=drive_link) |
| CMS Hospital Readmissions Reduction Program Overview | Explains the policy context and financial stakes of readmissions | [link](https://www.cms.gov/medicare/payment/prospective-payment-systems/acute-inpatient-pps/hospital-readmissions-reduction-program-hrrp) |
| Predicting Hospital Readmission in Diabetic Patients - a Review | Surveys ML appraches for readmission prediction | [link](https://drive.google.com/file/d/1VN6H53uJuL0AwqJDLV44tqkjNSHrIvaT/view?usp=drive_link) |
| ADA Standards of Medical Care in Diabetes | Clinical guidelines context for what "good" diabetes management looks like | [link](https://drive.google.com/file/d/19WEBjxOX3cVzF5qWQHgh0_zZ98VHjvHb/view?usp=drive_link) |
| Predicting and Preventing Acute Care Re-Utilization by Patients with Diabetes (Rubin & Shah, 2021) | Provides a review, summary and evaulation of possible interventions | [link](https://drive.google.com/file/d/18n2N2hcxbCOyAyPLZu7RVZXzjnMl_-WP/view?usp=drive_link)

## Data Creation

Bias Identification:

There are a few sources of bias in this dataset. Firstly, thre is a demographic representation bias as the dataset is drawn from 130 US hospitals, so it doesn't represent rural hospital, critical access hospital, or non-US health systems. Also it the distribution of demographic data is a reflection of the hospital mix of 1999-2008. Secoundly, there is a survivorship bias in the `readmitted` label, as it only captures readmissions in the same hospital and does not account for patients who died shortly after discharge. Thirdly, there is three features with high missingness, icnluding `weight`, which is likely correlated with the outcomes as it is related to metabolic concerns. Fourthly, there is a historical bias present because this dataset is from 1999-2008, many practices and policies of medicine have changed since then.

Bias Mitigation:

To attempt to address these potential sources of bias, I will employ the following mitigation strategies. Firstly, the demographic representation bias will be acknowledged and results will be reported stratified by race and age group, not generalizable beyond the original hosptial populaiton. Secondly, survivorship bias in the `readmitted` label cannot be altered so it will be contintually documented as a limitation. Thirdly, the missing data in weight will be mitigated by treating `weight` missingness as an indicator feature `weight_missing`. Fourthly, the historial biases will be continually addressed in the discussion section and any deployment of the model would require retraining on contemporary data.

Rationale: 

There are a few cirtical decisions that were made. Firstly, I binarized the target variable to fit wiht the project problem statment that is a binary classfiication task. Secondly. I dropped the high-missingness columns as a decision to avoid introducing more noise than signal.

There is uncertainly introduced by the feature engineering as the ICD-9 grouping schema might cause different results as grouping schemas have that possibility.
