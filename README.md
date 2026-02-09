# FUTURE_ML_02

# Support Ticket Classification System

An automated IT support ticket classification system using Machine Learning. Classifies tickets into predefined categories and assigns priority levels using NLP and business rules.

## Features

- Automatic ticket classification into categories (Hardware, Access, HR Support, etc.)
- Priority assignment (High, Medium, Low) based on ticket content
- Multiple ML models with performance comparison
- Text preprocessing with TF-IDF vectorization
- Confidence score calculation for predictions
- Model evaluation with detailed metrics

## Quick Start

### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook or Google Colab

### Installation

```bash
git clone https://github.com/yourusername/support-ticket-classifier.git
cd support-ticket-classifier
pip install -r requirements.txt
jupyter notebook Support_ticket_classification.ipynb
```

## Dependencies

```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

## Usage

### 1. Data Loading

Load your preprocessed ticket data in CSV format:
```python
df = pd.read_csv("all_tickets_processed_improved_v3.csv.zip")
```

Required columns:
- `Document`: Preprocessed ticket text
- `Topic_group`: Category label
- `Priority`: Priority level (High/Medium/Low)

### 2. Model Training

The notebook trains multiple models:
- Multinomial Naive Bayes
- Logistic Regression

Both models use TF-IDF vectorization for text feature extraction.

### 3. Prediction

Classify new tickets using the trained model:
```python
ticket_text = "My laptop won't turn on and I need it for work urgently"
category, confidence, priority = classify_ticket(ticket_text)
```

Output:
- Predicted Category (e.g., Hardware, Access, HR Support)
- Confidence Score (0-100%)
- Priority Level (High, Medium, Low)

## Model Performance

### Category Classification
- Accuracy: 84-86% (varies by model)
- Models tested: Multinomial NB, Logistic Regression
- Feature extraction: TF-IDF

### Priority Prediction
- Accuracy: 81.49%
- Model: Multinomial Naive Bayes
- Classes: High (84% F1), Medium (76% F1), Low (80% F1)

## Workflow

1. **Data Exploration**
   - Load preprocessed ticket data
   - Analyze category distribution
   - Explore text patterns

2. **Feature Engineering**
   - TF-IDF vectorization of ticket text
   - Max features: configurable
   - N-gram range: unigrams and bigrams

3. **Model Training**
   - Train/test split (80/20)
   - Stratified sampling for balanced classes
   - Multiple model comparison

4. **Priority Assignment**
   - Separate model for priority prediction
   - Based on ticket content and context
   - Three levels: High, Medium, Low

5. **Evaluation**
   - Classification reports
   - Confusion matrices
   - Accuracy, Precision, Recall, F1-score

6. **Prediction**
   - Classify new incoming tickets
   - Return category, confidence, and priority
   - Real-time inference capability

## Categories Supported

- Hardware (laptop issues, equipment failures)
- Access (password resets, permissions, login issues)
- HR Support (employee requests, administrative tasks)
- Miscellaneous (other support requests)

## Business Applications

**IT Support Teams**
- Automatic ticket routing to specialized teams
- Faster response times through priority assignment
- Reduced manual categorization effort

**Service Desk Optimization**
- 24/7 automatic ticket classification
- Consistent categorization standards
- Performance metrics tracking

**Customer Experience**
- Quicker ticket resolution
- Better resource allocation
- Improved SLA compliance

## Technical Details

### Text Processing
- Preprocessing: Tokenization, stop word removal, stemming
- Vectorization: TF-IDF with configurable parameters
- Feature selection: Most informative terms

### Classification Approach
- Supervised learning with labeled training data
- Multi-class classification problem
- Probability-based confidence scoring

### Model Selection
- Naive Bayes: Fast, efficient for text classification
- Logistic Regression: Better accuracy, interpretable results
- Choose based on accuracy vs speed requirements

## Performance Metrics

| Metric | Value |
|--------|-------|
| Category Accuracy | 84-86% |
| Priority Accuracy | 81.49% |
| Training Time | 2-5 seconds |
| Prediction Time | <100ms per ticket |
| F1-Score (weighted avg) | 0.81-0.84 |

## Example Predictions

**Input**: "My laptop won't turn on and I need it for work urgently"
- Category: Hardware
- Confidence: 63.75%
- Priority: High

**Input**: "I forgot my password and can't login to my account"
- Category: Access
- Confidence: 96.70%
- Priority: High

**Input**: "I need access to the shared drive for the marketing folder"
- Category: HR Support
- Confidence: 41.25%
- Priority: Medium

## Data Requirements

Training data should include:
- Minimum 1000 tickets per category
- Balanced class distribution (or use stratified sampling)
- Preprocessed text (cleaned, tokenized)
- Labeled categories and priorities
- Representative of production tickets

## Acknowledgments

Built with scikit-learn, pandas, and matplotlib for text classification and NLP applications.
Designed for IT support teams to automate ticket classification and improve response efficiency.
