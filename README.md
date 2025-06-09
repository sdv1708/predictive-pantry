# Instacart Market Basket Analysis & Reorder Prediction  
**Author**: Sanjay DV

---

## Project Overview

This project is a deep dive into the Instacart Market Basket Analysis dataset. The primary goal is to understand customer purchasing behavior and to build a predictive model that can accurately forecast which of a user's previously purchased products they will reorder in their next session.

By predicting reorders, Instacart can:
- Enhance the user experience by pre-populating shopping carts
- Improve its recommendation engine
- Optimize inventory and supply chain logistics for retail partners

---

## The Business Problem

In the competitive landscape of online grocery delivery, customer retention and satisfaction are paramount. A key challenge is to make the shopping experience as seamless and personalized as possible. Predicting which items a customer will buy again addresses several key business objectives:

- **Enhance User Experience**: Save customers time and effort by suggesting frequently reordered items.
- **Increase Sales**: Personalized reorder predictions can increase basket size.
- **Optimize Inventory**: Predictive insights help manage stock levels and reduce spoilage.
- **Targeted Marketing**: Understand reorder drivers to create effective promotions.

---

## About the Dataset

This project uses the **Instacart Market Basket Analysis** dataset made publicly available on Kaggle. The anonymized dataset includes over **3 million grocery orders** from more than **200,000 users** and is split across several relational CSV files:

| File | Description |
|------|-------------|
| `orders.csv` | Order-level details including user ID, order sequence, and timing |
| `products.csv` | Product names and identifiers |
| `aisles.csv` | Aisle-level categorization |
| `departments.csv` | Department-level categorization |
| `order_products__prior.csv` | Contents of prior orders |
| `order_products__train.csv` | Training data for reorder prediction |

---

## Tech Stack

- **Data Manipulation & Analysis**: Pandas, NumPy  
- **Data Visualization**: Matplotlib, Seaborn  
- **Machine Learning**: Scikit-learn  
- **Model Interpretability**: SHAP  
- **Development Environment**: Jupyter Notebook, Python 3

---

## 🔬 Methodology

### Step 1: Exploratory Data Analysis (EDA)
- **Order timing**: Heatmaps show peak order times are weekends and midday hours.
- **Popular products**: Fresh produce dominates top-selling categories.
- **Reorder behavior**: Most users reorder weekly or monthly.

### Step 2: Feature Engineering & Reordering Analysis
- **Reorder ratios**: High in dairy, snacks, and beverages.
- **Add-to-cart order**: Strong predictor—items added earlier are more likely reorders.

### Step 3: Predictive Modeling
Models used:

- **Logistic Regression**: Interpretable baseline.
- **Decision Tree Classifier**: Captures non-linear rules; pruned for generalization.
- **Ensemble Models**:
  - **Random Forest**: Good for reducing overfitting.
  - **Gradient Boosting**: Best performance through sequential learning and SHAP explanations.

---

## Model Performance & Evaluation

| Model                | Accuracy | Precision | Recall  | F1-Score | ROC-AUC |
|----------------------|----------|-----------|---------|----------|---------|
| Logistic Regression  | 70.27%   | 72.62%    | 79.98%  | 0.76     | 0.75    |
| Decision Tree (Opt.) | 65.76%   | 67.00%    | 90.00%  | 0.77     | —       |
| Gradient Boosting    | —        | —         | —       | —        | Best    |

**Why these metrics matter**:
- **Recall**: Key for capturing actual reorders — missing a reorder is a worse experience than suggesting a wrong item.
- **Precision**: Ensures suggested reorders are relevant.
- **ROC-AUC**: Measures model's overall classification ability across thresholds.

---

## Key Insights & Feature Importance

Using SHAP on the Gradient Boosting model, key predictors were identified:

- **`add_to_cart_order`**: Top predictor. Items added early are highly likely to be reorders.
- **`days_since_prior_order`**: Shorter gaps indicate habitual buying.
- **Department/Aisle**: Reorder likelihood tied to essentials like dairy, pantry, and produce.

---

## Actionable Business Recommendations

- **Inventory Management**: Ensure availability for high-reorder items (milk, snacks, produce).
- **Personalized Marketing**:
  - Smart Reorder prompts based on top reorders
  - Bundle offers for frequently co-purchased items
  - Discounts on low-reorder items to increase trial
- **Customer Retention**: Design loyalty programs based on reorder patterns

---

## Conclusion & Future Work

This project delivers an end-to-end data science solution to a real-world business challenge. By understanding reorder behavior, Instacart can:

- Enhance UX via personalized pre-filled carts
- Reduce churn
- Improve inventory and marketing strategy

**Future directions**:
- Deploy the model as a real-time API
- Develop user-specific features (e.g., personal reorder rate)
- Explore deep learning approaches

---

## How to Run the Code

1. **Clone the repository**:
   ```bash
   git clone https://github.com/sdv1708/predictive-pantry.git
   cd instacart-reorder-prediction

2. Download the dataset from the Kaggle competition page.

3. Place all .csv files inside a data/ directory in your project root.

4. Install dependencies: 
- pip install -r requirements.txt

5. Launch the notebook on your preferred IDE or cloud platform (Colab, Jupyter .etc)
