```python
# Just setting up the environment
import os
import logging
import warnings
import pandas as pd
import numpy as np
import duckdb
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score,
    ConfusionMatrixDisplay
)
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import GridSearchCV

from google.colab import drive
drive.mount('/content/drive')

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s — %(levelname)s — %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

plt.style.use('seaborn-v0_8-whitegrid')
PALETTE   = ['#2E86AB', '#E84855', '#F9C22E', '#3BB273', '#7B2D8B']
FIG_DPI   = 150
FIG_DIR   = 'figures/'
os.makedirs(FIG_DIR, exist_ok=True)

RANDOM_STATE = 42

logger.info('Environment configured successfully.')
```

    Mounted at /content/drive



```python
#getting the files
DATA_DIR = '/content/drive/MyDrive/26 Spring Sem/DS4320/Project 1/'

FILES = {
    'diabetic_readmission_cleaned': os.path.join(DATA_DIR, 'diabetic_readmission_cleaned.csv'),
    'medicines'                   : os.path.join(DATA_DIR, 'medicines.csv'),
    'admission_type_id'           : os.path.join(DATA_DIR, 'admission_type_id.csv'),
    'discharge_disposition_id'    : os.path.join(DATA_DIR, 'discharge_disposition_id.csv'),
    'admission_source_id'         : os.path.join(DATA_DIR, 'admission_source_id.csv'),
}

# Checking that i do have all the files
missing = [name for name, path in FILES.items() if not os.path.exists(path)]
if missing:
    raise FileNotFoundError(
        f"The following data files were not found: {missing}.\n"
        f"Please update DATA_DIR to point to your local data folder."
    )
logger.info('All data files located successfully.')

# Create an in-memory DuckDB connection
con = duckdb.connect(database=':memory:')

# loading each data file as a DuckDB table
for table_name, filepath in FILES.items():
    try:
        type_override = ""
        # overriding medicines table columns to be varchar and not boolean (except encounter_id)
        if table_name == 'medicines':
            df_header = pd.read_csv(filepath, nrows=0)

            medicine_columns_to_varchar = [col for col in df_header.columns if col != 'encounter_id']

            types_dict = {col: 'VARCHAR' for col in medicine_columns_to_varchar}

            if types_dict:
                type_override = f", types={types_dict}"

        con.execute(f"""
            CREATE OR REPLACE TABLE {table_name} AS
            SELECT * FROM read_csv_auto('{filepath}'{type_override})
        """)
        row_count = con.execute(f'SELECT COUNT(*) FROM {table_name}').fetchone()[0]
        logger.info(f'Loaded table [{table_name}] — {row_count:,} rows.')
    except Exception as e:
        logger.error(f'Failed to load {table_name}: {e}')
        raise

logger.info('All tables loaded into DuckDB.')
```


    FloatProgress(value=0.0, layout=Layout(width='auto'), style=ProgressStyle(bar_color='black'))



```python
# query: replacing the numeric codes with descriptive labels so that it the admission_type, discharge_disposition, and admission_source are more easily interpretable

query_human_readable = """
SELECT
    d.encounter_id,
    d.patient_nbr,
    atype.description AS admission_type,
    dd.description AS discharge_disposition,
    src.description AS admission_source,
    d.time_in_hospital,
    d.num_lab_procedures,
    d.num_procedures,
    d.num_medications,
    d.number_outpatient,
    d.number_emergency,
    d.number_inpatient,
    d.number_diagnoses,
    d.weight_missing,
    d.readmitted_30d
FROM diabetic_readmission_cleaned d
LEFT JOIN admission_type_id atype
    ON d.admission_type_id = atype.admission_type_id
LEFT JOIN discharge_disposition_id dd
    ON d.discharge_disposition_id = dd.discharge_disposition_id
LEFT JOIN admission_source_id src
    ON d.admission_source_id = src.admission_source_id
LIMIT 10
"""

df_human = con.execute(query_human_readable).df()
logger.info(f'Query 1 (human-readable) returned {len(df_human)} preview rows.')
print('=== Query 1: Human-Readable Encounter Summary (first 10 rows) ===')
df_human
```

    === Query 1: Human-Readable Encounter Summary (first 10 rows) ===






  <div id="df-2af4d584-65f7-4da3-9113-ddaeeb176b75" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>encounter_id</th>
      <th>patient_nbr</th>
      <th>admission_type</th>
      <th>discharge_disposition</th>
      <th>admission_source</th>
      <th>time_in_hospital</th>
      <th>num_lab_procedures</th>
      <th>num_procedures</th>
      <th>num_medications</th>
      <th>number_outpatient</th>
      <th>number_emergency</th>
      <th>number_inpatient</th>
      <th>number_diagnoses</th>
      <th>weight_missing</th>
      <th>readmitted_30d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2278392</td>
      <td>8222157</td>
      <td>NULL</td>
      <td>Expired, place unknown. Medicaid only, hospice.</td>
      <td>Physician Referral</td>
      <td>1</td>
      <td>41</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>149190</td>
      <td>55629189</td>
      <td>Emergency</td>
      <td>Discharged to home</td>
      <td>Emergency Room</td>
      <td>3</td>
      <td>59</td>
      <td>0</td>
      <td>18</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>64410</td>
      <td>86047875</td>
      <td>Emergency</td>
      <td>Discharged to home</td>
      <td>Emergency Room</td>
      <td>2</td>
      <td>11</td>
      <td>5</td>
      <td>13</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>6</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>500364</td>
      <td>82442376</td>
      <td>Emergency</td>
      <td>Discharged to home</td>
      <td>Emergency Room</td>
      <td>2</td>
      <td>44</td>
      <td>1</td>
      <td>16</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>16680</td>
      <td>42519267</td>
      <td>Emergency</td>
      <td>Discharged to home</td>
      <td>Emergency Room</td>
      <td>1</td>
      <td>51</td>
      <td>0</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>35754</td>
      <td>82637451</td>
      <td>Urgent</td>
      <td>Discharged to home</td>
      <td>Clinic Referral</td>
      <td>3</td>
      <td>31</td>
      <td>6</td>
      <td>16</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>55842</td>
      <td>84259809</td>
      <td>Elective</td>
      <td>Discharged to home</td>
      <td>Clinic Referral</td>
      <td>4</td>
      <td>70</td>
      <td>1</td>
      <td>21</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>63768</td>
      <td>114882984</td>
      <td>Emergency</td>
      <td>Discharged to home</td>
      <td>Emergency Room</td>
      <td>5</td>
      <td>73</td>
      <td>0</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>12522</td>
      <td>48330783</td>
      <td>Urgent</td>
      <td>Discharged to home</td>
      <td>Transfer from a hospital</td>
      <td>13</td>
      <td>68</td>
      <td>2</td>
      <td>28</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>15738</td>
      <td>63555939</td>
      <td>Elective</td>
      <td>Discharged/transferred to SNF</td>
      <td>Transfer from a hospital</td>
      <td>12</td>
      <td>33</td>
      <td>3</td>
      <td>18</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-2af4d584-65f7-4da3-9113-ddaeeb176b75')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-2af4d584-65f7-4da3-9113-ddaeeb176b75 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-2af4d584-65f7-4da3-9113-ddaeeb176b75');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


  <div id="id_19d4ffd6-c69a-4719-8e08-9bf635b16b6e">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('df_human')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_19d4ffd6-c69a-4719-8e08-9bf635b16b6e button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('df_human');
      }
      })();
    </script>
  </div>

    </div>
  </div>





```python
#query: figuring out what admission type has the highest readmission rate

query_by_admission = """
SELECT
    atype.description AS admission_type,
    COUNT(*) AS n_encounters,
    SUM(d.readmitted_30d) AS n_readmitted,
    ROUND(100.0 * SUM(d.readmitted_30d)
          / COUNT(*), 2) AS readmission_rate_pct
FROM diabetic_readmission_cleaned d
LEFT JOIN admission_type_id atype
       ON d.admission_type_id = CAST(atype.admission_type_id AS INTEGER)
GROUP BY atype.description
ORDER BY readmission_rate_pct DESC
"""

df_by_admission = con.execute(query_by_admission).df()
logger.info('Query 2 (readmission by admission type) executed.')
print('=== Query 2: Readmission Rate by Admission Type ===')
df_by_admission
```

    === Query 2: Readmission Rate by Admission Type ===






  <div id="df-740e9ca2-4a1d-4ac8-976f-23539613eb03" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>admission_type</th>
      <th>n_encounters</th>
      <th>n_readmitted</th>
      <th>readmission_rate_pct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Emergency</td>
      <td>53990</td>
      <td>6221.0</td>
      <td>11.52</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Urgent</td>
      <td>18480</td>
      <td>2066.0</td>
      <td>11.18</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NULL</td>
      <td>5291</td>
      <td>586.0</td>
      <td>11.08</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Elective</td>
      <td>18869</td>
      <td>1961.0</td>
      <td>10.39</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Not Available</td>
      <td>4785</td>
      <td>495.0</td>
      <td>10.34</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Newborn</td>
      <td>10</td>
      <td>1.0</td>
      <td>10.00</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Not Mapped</td>
      <td>320</td>
      <td>27.0</td>
      <td>8.44</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Trauma Center</td>
      <td>21</td>
      <td>0.0</td>
      <td>0.00</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-740e9ca2-4a1d-4ac8-976f-23539613eb03')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-740e9ca2-4a1d-4ac8-976f-23539613eb03 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-740e9ca2-4a1d-4ac8-976f-23539613eb03');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


  <div id="id_722d42f9-00b9-404e-a5cb-a538de08ad5a">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('df_by_admission')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_722d42f9-00b9-404e-a5cb-a538de08ad5a button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('df_by_admission');
      }
      })();
    </script>
  </div>

    </div>
  </div>





```python
# query: seeing if a change in insulin regimen at discharge means there is a lower readmission rate

query_insulin = """
SELECT
    insulin_status,
    COUNT(*) AS n_encounters,
    SUM(readmitted_30d) AS n_readmitted,
    ROUND(100.0 * SUM(readmitted_30d)
          / COUNT(*), 2) AS readmission_rate_pct
FROM (
    -- Derive insulin status label from the insulin column in the medicines table.
    -- The medicines table has a single 'insulin' column with values like 'Up', 'Down', 'Steady', 'No'.
    SELECT
        d.readmitted_30d,
        m.insulin AS insulin_status
    FROM diabetic_readmission_cleaned d
    JOIN medicines m
      ON d.encounter_id = m.encounter_id
) sub
GROUP BY insulin_status
ORDER BY readmission_rate_pct DESC
"""

df_insulin = con.execute(query_insulin).df()
logger.info('Query 3 (insulin vs readmission) executed.')
print('=== Query 3: Readmission Rate by Insulin Status ===')
df_insulin
```

    === Query 3: Readmission Rate by Insulin Status ===






  <div id="df-3c52f5e7-c8ee-49e8-baf7-8dbc3fea48d2" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>insulin_status</th>
      <th>n_encounters</th>
      <th>n_readmitted</th>
      <th>readmission_rate_pct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Down</td>
      <td>12218</td>
      <td>1698.0</td>
      <td>13.90</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Up</td>
      <td>11316</td>
      <td>1470.0</td>
      <td>12.99</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Steady</td>
      <td>30849</td>
      <td>3433.0</td>
      <td>11.13</td>
    </tr>
    <tr>
      <th>3</th>
      <td>No</td>
      <td>47383</td>
      <td>4756.0</td>
      <td>10.04</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-3c52f5e7-c8ee-49e8-baf7-8dbc3fea48d2')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-3c52f5e7-c8ee-49e8-baf7-8dbc3fea48d2 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-3c52f5e7-c8ee-49e8-baf7-8dbc3fea48d2');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


  <div id="id_a9a4d345-a7b0-4ee9-885b-0e1064037b62">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('df_insulin')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_a9a4d345-a7b0-4ee9-885b-0e1064037b62 button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('df_insulin');
      }
      })();
    </script>
  </div>

    </div>
  </div>





```python
#query: pulling the dataset made in duckdb into pandas df

query_model_data = """
SELECT * FROM diabetic_readmission_cleaned
"""

df_model = con.execute(query_model_data).df()
logger.info(f'Model dataset pulled from DuckDB: {df_model.shape[0]:,} rows × {df_model.shape[1]} cols.')
print(f'Model DataFrame shape: {df_model.shape}')
print(f'Target class distribution:')
print(df_model['readmitted_30d'].value_counts(normalize=True).rename({0:'Not readmitted',1:'Readmitted <30d'}).round(4))
```

    Model DataFrame shape: (101766, 2275)
    Target class distribution:
    readmitted_30d
    Not readmitted     0.8884
    Readmitted <30d    0.1116
    Name: proportion, dtype: float64



```python
#eda...yay...

numeric_features = [
    'time_in_hospital', 'num_lab_procedures', 'num_procedures',
    'num_medications', 'number_outpatient', 'number_emergency',
    'number_inpatient', 'number_diagnoses'
]

fig = plt.figure(figsize=(16, 12), dpi=FIG_DPI)
fig.suptitle(
    'Diabetic Readmission Dataset — Exploratory Overview',
    fontsize=16, fontweight='bold', y=0.98
)
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

# pie chart of readmitted/not readmitted
ax_pie = fig.add_subplot(gs[0, :2])
counts = df_model['readmitted_30d'].value_counts()
labels = ['Not Readmitted\n(<30d)', 'Readmitted\nwithin 30d']
ax_pie.pie(
    counts, labels=labels, colors=[PALETTE[0], PALETTE[1]],
    autopct='%1.1f%%', startangle=90,
    wedgeprops={'edgecolor': 'white', 'linewidth': 1.5},
    textprops={'fontsize': 11}
)
ax_pie.set_title('Target Class Distribution', fontsize=12, fontweight='bold')

# readmission rate by number of inpatient visits
ax_inp = fig.add_subplot(gs[0, 2:])
inp_rate = (
    df_model.groupby('number_inpatient')['readmitted_30d']
    .mean().reset_index()
    .rename(columns={'readmitted_30d': 'readmission_rate'})
    .query('number_inpatient <= 10')  # Cap for readability
)
ax_inp.bar(
    inp_rate['number_inpatient'], inp_rate['readmission_rate'],
    color=PALETTE[2], edgecolor='white'
)
ax_inp.set_xlabel('Prior Inpatient Visits', fontsize=10)
ax_inp.set_ylabel('30-day Readmission Rate', fontsize=10)
ax_inp.set_title('Readmission Rate vs Prior Inpatient Visits', fontsize=11, fontweight='bold')
ax_inp.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))

# histograms of key numeric features split by readmission label
for i, feat in enumerate(numeric_features):
    row, col = divmod(i, 4)
    ax = fig.add_subplot(gs[row + 1, col])
    for label, color in zip([0, 1], [PALETTE[0], PALETTE[1]]):
        subset = df_model.loc[df_model['readmitted_30d'] == label, feat]
        ax.hist(
            subset, bins=25, alpha=0.6,
            color=color, edgecolor='none', density=True
        )
    ax.set_title(feat.replace('_', ' ').title(), fontsize=9, fontweight='bold')
    ax.set_xlabel('')
    ax.tick_params(labelsize=7)

# Shared legend for histogram panels
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=PALETTE[0], alpha=0.7, label='Not Readmitted'),
    Patch(facecolor=PALETTE[1], alpha=0.7, label='Readmitted <30d')
]
fig.legend(handles=legend_elements, loc='lower center', ncol=2,
           fontsize=10, frameon=True, bbox_to_anchor=(0.5, -0.01))

plt.savefig(os.path.join(FIG_DIR, 'eda_overview.png'), bbox_inches='tight', dpi=FIG_DPI)
plt.show()
logger.info('EDA overview figure saved.')
```


    
![png](Project1_pipeline_files/Project1_pipeline_6_0.png)
    



```python
#preparing the features

# Dropping the identifier columns b/c they have no predictive value
DROP_COLS = ['encounter_id', 'patient_nbr']
TARGET    = 'readmitted_30d'

feature_cols = [c for c in df_model.columns if c not in DROP_COLS + [TARGET]]
X = df_model[feature_cols].copy()
y = df_model[TARGET].copy()

# making sure all columns are numeric (fillna with 0 as a safety net)
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

logger.info(f'Feature matrix shape: {X.shape}')
logger.info(f'Target distribution — 0: {(y==0).sum():,}  |  1: {(y==1).sum():,}')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)
logger.info(f'Train size: {len(X_train):,}  |  Test size: {len(X_test):,}')

print(f'Training set: {X_train.shape[0]:,} rows | Test set: {X_test.shape[0]:,} rows')
print(f'Features used: {X_train.shape[1]}')
```

    Training set: 81,412 rows | Test set: 20,354 rows
    Features used: 2272



```python
# modeling as a logistic regression...let's see how this goes
#doing this because this is the interpretable option for binary classification

lr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        solver='lbfgs',
        random_state=RANDOM_STATE
    ))
])

try:
    lr_pipeline.fit(X_train, y_train)
    y_pred_lr   = lr_pipeline.predict(X_test)
    y_proba_lr  = lr_pipeline.predict_proba(X_test)[:, 1]
    auc_lr      = roc_auc_score(y_test, y_proba_lr)
    ap_lr       = average_precision_score(y_test, y_proba_lr)
    logger.info(f'Logistic Regression — AUC-ROC: {auc_lr:.4f} | Avg Precision: {ap_lr:.4f}')
    print('=== Logistic Regression Classification Report ===')
    print(classification_report(y_test, y_pred_lr, target_names=['Not Readmitted', 'Readmitted <30d']))
    print(f'AUC-ROC: {auc_lr:.4f}  |  Average Precision: {ap_lr:.4f}')
except Exception as e:
    logger.error(f'Logistic Regression training failed: {e}')
    raise
```

    === Logistic Regression Classification Report ===
                     precision    recall  f1-score   support
    
     Not Readmitted       0.92      0.65      0.76     18083
    Readmitted <30d       0.16      0.55      0.25      2271
    
           accuracy                           0.64     20354
          macro avg       0.54      0.60      0.51     20354
       weighted avg       0.84      0.64      0.71     20354
    
    AUC-ROC: 0.6356  |  Average Precision: 0.1933



```python
# modeling as a random forest classifier b/c might do better if the relationship is non-linear

# defining the parameter grid for the Random Forest Classifier
param_grid = {
    'n_estimators': [100, 150, 200],
    'max_depth': [10, 12, 15],
    'min_samples_leaf': [10, 15, 20]
}

# initizalizing the Random Forest Classifier
rf_base_model = RandomForestClassifier(
    class_weight='balanced',
    random_state=RANDOM_STATE,
    n_jobs=-1
)

# initizalizing GridSearchCV
grid_search = GridSearchCV(
    estimator=rf_base_model,
    param_grid=param_grid,
    scoring='roc_auc',
    cv=3,
    n_jobs=-1,
    verbose=1
)

try:
    logger.info('Starting Random Forest hyperparameter tuning...')
    grid_search.fit(X_train, y_train)

    best_rf_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    logger.info(f'Best Random Forest parameters: {best_params}')
    logger.info(f'Best cross-validation AUC-ROC score: {best_score:.4f}')

    y_pred_rf = best_rf_model.predict(X_test)
    y_proba_rf = best_rf_model.predict_proba(X_test)[:, 1]
    auc_rf = roc_auc_score(y_test, y_proba_rf)
    ap_rf = average_precision_score(y_test, y_proba_rf)
    logger.info(f'Best Random Forest Model (Test Set) — AUC-ROC: {auc_rf:.4f} | Avg Precision: {ap_rf:.4f}')

    print('=== Random Forest Hyperparameter Tuning Results ===')
    print(f'Best parameters found: {best_params}')
    print(f'Best cross-validation AUC-ROC score: {best_score:.4f}')
    print('\n=== Best Random Forest Model Classification Report (Test Set) ===')
    print(classification_report(y_test, y_pred_rf, target_names=['Not Readmitted', 'Readmitted <30d']))
    print(f'AUC-ROC: {auc_rf:.4f}  |  Average Precision: {ap_rf:.4f}')

except Exception as e:
    logger.error(f'Random Forest hyperparameter tuning or training failed: {e}')
    raise

```

    Fitting 3 folds for each of 27 candidates, totalling 81 fits
    === Random Forest Hyperparameter Tuning Results ===
    Best parameters found: {'max_depth': 15, 'min_samples_leaf': 10, 'n_estimators': 200}
    Best cross-validation AUC-ROC score: 0.6478
    
    === Best Random Forest Model Classification Report (Test Set) ===
                     precision    recall  f1-score   support
    
     Not Readmitted       0.93      0.62      0.74     18083
    Readmitted <30d       0.17      0.61      0.26      2271
    
           accuracy                           0.62     20354
          macro avg       0.55      0.61      0.50     20354
       weighted avg       0.84      0.62      0.69     20354
    
    AUC-ROC: 0.6581  |  Average Precision: 0.2033



```python
# Performance Summary Table of the two different models

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

results = []
for name, y_pred, y_proba in [
    ('Logistic Regression', y_pred_lr, y_proba_lr),
    ('Random Forest',       y_pred_rf, y_proba_rf),
]:
    results.append({
        'Model'           : name,
        'Accuracy'        : round(accuracy_score(y_test, y_pred), 4),
        'Precision (pos)' : round(precision_score(y_test, y_pred), 4),
        'Recall (pos)'    : round(recall_score(y_test, y_pred), 4),
        'F1 (pos)'        : round(f1_score(y_test, y_pred), 4),
        'AUC-ROC'         : round(roc_auc_score(y_test, y_proba), 4),
        'Avg Precision'   : round(average_precision_score(y_test, y_proba), 4),
    })

summary_df = pd.DataFrame(results).set_index('Model')
logger.info('Performance summary computed.')
print('Model Performance Summary')
summary_df
```

    Model Performance Summary






  <div id="df-cf133f07-e511-4607-bf8d-2a1ef603c470" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Accuracy</th>
      <th>Precision (pos)</th>
      <th>Recall (pos)</th>
      <th>F1 (pos)</th>
      <th>AUC-ROC</th>
      <th>Avg Precision</th>
    </tr>
    <tr>
      <th>Model</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Logistic Regression</th>
      <td>0.6403</td>
      <td>0.1649</td>
      <td>0.5469</td>
      <td>0.2533</td>
      <td>0.6356</td>
      <td>0.1933</td>
    </tr>
    <tr>
      <th>Random Forest</th>
      <td>0.6204</td>
      <td>0.1679</td>
      <td>0.6072</td>
      <td>0.2630</td>
      <td>0.6581</td>
      <td>0.2033</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-cf133f07-e511-4607-bf8d-2a1ef603c470')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-cf133f07-e511-4607-bf8d-2a1ef603c470 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-cf133f07-e511-4607-bf8d-2a1ef603c470');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


  <div id="id_7926780b-cff7-42e7-86a8-7475f29883ca">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('summary_df')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_7926780b-cff7-42e7-86a8-7475f29883ca button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('summary_df');
      }
      })();
    </script>
  </div>

    </div>
  </div>




**Analysis rationale:**

Initially I started with a logistic regression model, but I suspected that there might be a non-linear relationship between the variables which would make the logisitc regression model not the best choice. Going off this hunch, I tried a random forest model. Now the Random Forest model technically had a lower accuracy than the logistic regression, but it did have a higher recall, which means that there were less people slipping through who would be readmitted and more false positive flags for readmittance. Considering the domain and the risks of not flagging for readmittance, the random forest model is a better option as it would be better to prep more people than miss people who would need to then be readmitted.


```python
# Random Forest Feature Importance (Top 20) - using to indicate strongest predictors

importances = pd.Series(
    best_rf_model.feature_importances_,
    index=feature_cols
).sort_values(ascending=False)

top_n    = 20
top_feats = importances.head(top_n)

# highlighting a few known important clinical features
clinical_highlight = {'number_inpatient', 'time_in_hospital', 'num_medications',
                       'number_diagnoses', 'num_lab_procedures'}
bar_colors = [
    PALETTE[1] if f in clinical_highlight else PALETTE[0]
    for f in top_feats.index
]

fig, ax = plt.subplots(figsize=(10, 7), dpi=FIG_DPI)

bars = ax.barh(
    top_feats.index[::-1],
    top_feats.values[::-1],
    color=bar_colors[::-1],
    edgecolor='white'
)

ax.set_xlabel('Mean Decrease in Gini Impurity (Feature Importance)', fontsize=11)
ax.set_title(
    f'Top {top_n} Predictors of 30-Day Readmission\n(Random Forest, n = {len(X_train):,} training encounters)',
    fontsize=13, fontweight='bold'
)

for bar in bars:
    width = bar.get_width()
    ax.text(width + 0.0002, bar.get_y() + bar.get_height() / 2,
            f'{width:.4f}', va='center', ha='left', fontsize=8)

from matplotlib.patches import Patch
legend_elems = [
    Patch(color=PALETTE[1], label='Core clinical feature'),
    Patch(color=PALETTE[0], label='Encoded categorical feature')
]
ax.legend(handles=legend_elems, fontsize=10, loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'feature_importance.png'), bbox_inches='tight', dpi=FIG_DPI)
plt.show()
logger.info('Feature importance figure saved.')
```


    
![png](Project1_pipeline_files/Project1_pipeline_12_0.png)
    



```python
# Press Release / Publication Chart: Readmission Rate by Number of Inpatient Visits
# making a simple chart more suited for a press release considering non-domain people would be viewing

plot_df = inp_rate

fig, ax = plt.subplots(figsize=(10, 5), dpi=FIG_DPI)

bar_colors = [PALETTE[0]] * len(plot_df)

bars = ax.bar(
    plot_df['number_inpatient'],
    plot_df['readmission_rate'],
    color=bar_colors, edgecolor='white', width=0.65
)

ax.axhline(y=df_model['readmitted_30d'].mean(), color='gray',
           linestyle='--', lw=1.5, label='Overall average readmission rate')

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, height + 0.005,
            f'{height:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_xlabel('Prior Inpatient Visits', fontsize=11)
ax.set_ylabel('30-Day Readmission Rate (%)', fontsize=11)
ax.set_title(
    '30-Day Readmission Rates by Prior Inpatient Visits\nUCI Diabetes 130-US Hospitals Dataset',
    fontsize=13, fontweight='bold'
)
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))
ax.legend(fontsize=10)
ax.set_ylim(0, plot_df['readmission_rate'].max() * 1.18)
sns.despine(left=False, bottom=False)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'readmission_by_number_inpatient.png'), bbox_inches='tight', dpi=FIG_DPI)
plt.show()
logger.info('Readmission-by-number-inpatient chart saved.')

```


    
![png](Project1_pipeline_files/Project1_pipeline_13_0.png)
    


### Visualization Rationale

I did a few different visualizations messing around trying to determine what would be the best idea for the press release and comparing the two models more in depth. I choose to create a visualization of the feature importance for the random forest model to see exactly how the model was functioning and what is impacting the readmission results. Based on the visualization of the feature importance, I then made a visualization for the press release that displayed readmittance by the number of inpatient visits in the year prior as that was the most importnat feature. I choose to use this visualization for my press release because I think it is easily understandable for an audience who is not necessarily super familiar with the domain.
