# Fake News and Phishing Email Classification Projects

Two individual machine-learning notebook projects using text classification:

1. **Fake-news classification** — compares Fake and Real article labels using cleaned text and TF-IDF features, with additional data-quality controls to examine publisher and duplicate-text effects.
2. **Phishing-email classification** — predicts Phishing Email or Safe Email from email text and simple structural features.

Both notebooks are educational experiments on public Kaggle datasets. Their scores describe the supplied data and saved evaluation procedures; they are not evidence of reliable fact-checking or production email security.

## Contents

```text
.
├── README.md
├── requirements.txt
├── fakenews_detection.ipynb
├── phishingemail_detection.ipynb

```

Notebook input names are case-sensitive on some systems. Keep the CSV files beside their notebooks or update the notebook paths.

## Datasets

### Fake and Real News Dataset

- Kaggle: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
- Commonly associated with the ISOT Fake News dataset.
- Download the two files `Fake.csv` and `True.csv`.
- The supplied notebook reads `Fake.csv` and `True.csv`, labels the rows `fake` and `real`, then combines and shuffles them.
- In the recorded notebook data, the combined starting set has **44,898 articles**: **23,481 Fake** and **21,417 Real**.
- The files provide article title, body text, subject, and date. The controlled model uses article text; subject is excluded because subject categories in this dataset are strongly associated with labels.
- Dataset page lists the license as **CC BY-NC-SA 4.0**. Review the Kaggle terms before redistributing data or using it commercially. This repository should link to the dataset rather than republish the CSVs unless your intended use is allowed.

### Phishing Email Detection

- Kaggle: https://www.kaggle.com/datasets/subhajournal/phishingemails
- Download the file `Phishing_Email.csv`.
- The notebook expects the columns `Email Text` and `Email Type`; it drops `Unnamed: 0` if present.
- The supplied CSV contains **18,650 rows**. Sixteen rows with missing email text are removed, leaving **18,634 messages**: **11,322 Safe Email** and **7,312 Phishing Email**.
- The dataset is moderately imbalanced (approximately **60.8% Safe** and **39.2% Phishing**). The notebook uses a stratified split to retain similar proportions in train and test data.
- Check the Kaggle dataset page for its current license and redistribution terms before sharing the CSV.

## Environment setup

Use Python 3.10 or later. From the repository directory:

```bash
python -m venv .venv
```

Activate the environment, then install dependencies:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

Open the notebooks in Jupyter:

```bash
jupyter notebook
```

Run the notebook cells in order after placing the corresponding CSV file(s) beside the notebook. The notebooks include saved outputs, but rerunning them requires the source CSVs. Model training and the MLP may take longer than the data-inspection cells, depending on hardware and library versions.

## Project 1 — Fake-news classification

### Objective

Train and compare binary classifiers that predict the dataset-provided `fake` or `real` label from article text. The task is **dataset-label classification**, not independent verification of factual claims.

### Approach

1. Load `Fake.csv` and `True.csv`, assign labels, concatenate the rows, shuffle with random state 42, and inspect class and field distributions.
2. Lowercase the text, remove punctuation and digits, tokenize, remove a hand-written stopword list and discard tokens of two characters or fewer.
3. Build a manual Bag-of-Words representation with a 3,000-term vocabulary as a learning/EDA exercise. The controlled classifiers use up to **5,000 TF-IDF unigram features**.
4. Compare three protocols:
   - **Protocol A — original notebook run:** 44,898 articles, stratified 80/20 split (35,918 train / 8,980 test), random state 42. TF-IDF was fit before splitting, so this is not the most controlled estimate.
   - **Protocol B — data cleaning controls:** removes empty and duplicate article texts before evaluation. The reported Random Forest accuracy is 98.83%.
   - **Protocol C — publisher-marker controls:** further removes selected news-agency and web-publishing markers, then fits TF-IDF on training text only. The working set contains 38,644 articles and the held-out test set contains 7,729 articles.
5. Report accuracy, precision, recall, F1, confusion matrices and ROC-AUC; inspect class/source patterns and held-out errors.

### Models compared

- **K-Nearest Neighbors (k=9):** similarity-based baseline; distances can be less informative in sparse, high-dimensional TF-IDF space.
- **Logistic Regression (max_iter=1,000):** efficient linear baseline for sparse text features.
- **Random Forest (200 trees):** non-linear ensemble comparison.
- **MLP (one hidden layer of 100 units; up to 300 iterations):** a modest non-linear neural-network comparison, not an extensively tuned architecture.

### Recorded results

Protocol C metrics below use **Real** as the positive class.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | **97.84%** | 97.61% | 98.47% | 98.04% | 0.9973 |
| MLP | 97.77% | 97.72% | 98.23% | 97.98% | 0.9976 |
| Random Forest | 97.67% | 97.50% | 98.28% | 97.88% | 0.9972 |
| KNN | 88.48% | 90.34% | 88.46% | 89.39% | 0.9450 |

The original Protocol A Random Forest run recorded **99.71% accuracy**. That score is retained for comparison, not presented as the best-controlled result: it used TF-IDF fit before the split and was affected by dataset-quality/source cues. Protocol B Random Forest recorded 98.83%. Under Protocol C, Logistic Regression had the highest accuracy, while the differences among the top three models were small. Five-fold cross-validation was reported only for Logistic Regression: **97.96% mean accuracy (standard deviation 0.13 percentage points)**.

### Core Python modules and libraries

- `pandas` — CSV loading, tabular data handling, and inspection.
- `numpy` — numerical operations.
- `re`, `string`, `collections.Counter` — text cleanup and the manual vocabulary/Bag-of-Words demonstration (Python standard library).
- `scikit-learn` — `TfidfVectorizer`, `LabelEncoder`, `train_test_split`, KNN, Logistic Regression, Random Forest, MLP, and classification metrics.
- `matplotlib` and `seaborn` — plots, metric comparisons and confusion matrices.

## Project 2 — Phishing-email classification

### Objective

Compare classifiers that predict the supplied `Phishing Email` or `Safe Email` label from an email body. The notebook does not inspect real sender identity, headers, attachment contents, or whether a URL is malicious.

### Approach

1. Load `Phishing_Email.csv`; remove the optional `Unnamed: 0` index column and rows with missing email text.
2. Cap exceptionally long raw messages at 20,000 characters for processing.
3. Create a cleaned text representation by removing HTML tags, URLs and email addresses, then lowercasing and tokenizing.
4. Extract up to **5,000 TF-IDF unigram/bigram features** (with `min_df=3`).
5. Extract nine structural measurements from message text: URL count, URL presence, HTML-tag count, exclamation count, dollar-sign count, uppercase ratio, digit ratio, average word length, and word count.
6. Scale the structural features to non-negative values and combine them with the text matrix, giving **5,009 features** per message.
7. Make a stratified 80/20 split with random state 42: **14,907 training** and **3,727 test** messages.
8. Compare four models and inspect classification reports, confusion matrices, ROC curves, Random Forest feature importance and six manually written prediction examples.

### Models compared

- **Logistic Regression:** linear text-classification baseline.
- **Random Forest (200 trees):** non-linear ensemble comparison and feature-importance estimate.
- **Multinomial Naive Bayes:** lightweight probabilistic baseline; it assumes conditional independence of features and requires non-negative inputs.
- **MLP (one hidden layer of 100 units; up to 300 iterations):** basic non-linear comparison, not a tuned deep-learning system.

### Recorded results

The saved result table's default precision, recall and F1 treat **Safe Email** as the positive class because `LabelEncoder` encodes `Phishing Email=0` and `Safe Email=1`. Accuracy is class-independent.

| Model | Accuracy | Precision* | Recall* | F1* | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| MLP | **96.94%** | 98.51% | 96.42% | 97.46% | 0.996 |
| Random Forest | 96.49% | 97.30% | 96.91% | 97.10% | 0.995 |
| Logistic Regression | 96.38% | 97.38% | 96.64% | 97.01% | 0.995 |
| Multinomial Naive Bayes | 96.06% | 96.78% | 96.73% | 96.75% | 0.993 |

\*For the MLP specifically, when treating **Phishing Email** as positive, derived precision is **94.64%**, recall **97.74%**, and F1 **96.16%**. Its confusion matrix has 33 missed phishing messages (false negatives) and 81 safe messages flagged as phishing (false positives), among 1,462 phishing and 2,265 safe test examples.

The six manually written example messages demonstrate prediction behavior; they are **not** extra held-out test results and do not contribute to the reported accuracy.

### Core Python modules and libraries

- `pandas` and `numpy` — data handling and numerical operations.
- `re`, `string`, `collections.Counter` — text cleanup and token analysis (Python standard library).
- `scipy.sparse.hstack` and `csr_matrix` — combining sparse TF-IDF and structural feature matrices.
- `scikit-learn` — TF-IDF, label encoding, MinMax scaling, stratified splitting, four classifiers and evaluation metrics.
- `matplotlib` and `seaborn` — visual analysis.

## Limitations and responsible interpretation

### Fake-news project

- Dataset labels and source groups are confounded: real and fake articles can differ by publisher, formatting, topic and writing style. Removing known markers reduces but does not remove these signals.
- A high score means the classifier predicts the dataset labels on a particular split. It does not mean the model has verified a claim or detects misinformation from new publishers.
- The collection is historical and limited in source/topic/time coverage. Results may not generalize to current news or other regions and languages.
- Protocol C improves leakage control for TF-IDF and selected publisher markers, but it is not a source-separated or time-separated external evaluation. Hyperparameters were not comprehensively tuned; cross-validation was reported only for Logistic Regression.

### Phishing-email project

- TF-IDF and `MinMaxScaler` were fit before the train/test split in the supplied notebook. This lets test-set distribution information influence preprocessing and creates a leakage risk; recorded scores may be optimistic. A stronger rerun should split first and fit preprocessing inside a scikit-learn `Pipeline` or equivalent training-only workflow.
- Duplicate removal is not documented, so similar messages could appear in both sets.
- Email body text alone omits sender address, headers, domain reputation, link destinations, attachments and authentication results. Structural counts are weak cues, not security rules.
- The classes are moderately imbalanced; accuracy alone does not express the cost of missed phishing versus false alarms. Threshold choice requires separate validation and an explicit error-cost decision.
- Labels are accepted as supplied, and the model is not tested for new campaigns, time drift, obfuscation, multilingual email or independently collected data.

### Both projects

These are reproducible learning prototypes, not deployed services. Do not use them as sole grounds to block news, discipline a sender, or declare content true, false, safe or malicious. Practical use would require leakage-safe training, independent and current validation, privacy review, documented threshold choices, monitoring, and human oversight.

## Reproducibility notes

- Both notebooks use `random_state=42` for the reported stratified train/test split.
- Re-running may produce slightly different MLP or solver results across library versions, hardware and numerical backends.
- Preserve the original notebooks and their outputs if you need to show the recorded results; run a separate corrected copy when experimenting with leakage-safe preprocessing.
- For a portfolio repository, record the Python and library versions used for your final rerun, include source attribution, and avoid committing private, personal or licensed dataset files without permission.

## Suggested citation and attribution

- Fake and Real News Dataset by clmentbisaillon on Kaggle: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
- Phishing Email Detection by subhajournal on Kaggle: https://www.kaggle.com/datasets/subhajournal/phishingemails

Please follow each dataset's current license, attribution requirements and Kaggle terms.
