# PCA and Correlated Variables: To Remove or Not to Remove?

## The Interview Question
> **Circle K**: "You are given a dataset. The dataset contains many variables, some of which are highly correlated and you know about it. Your manager has asked you to run a PCA. Would you remove correlated variables and why?"

## Why This Question Matters

This question is a sophisticated test of your understanding of dimensionality reduction techniques and multicollinearity handling. Companies like Circle K ask this because:

- **Tests Fundamental Understanding**: It reveals whether you understand PCA's core purpose and mechanism
- **Assesses Problem-Solving Logic**: Shows how you approach data preprocessing decisions
- **Evaluates Practical Experience**: Demonstrates whether you've actually worked with high-dimensional, correlated data
- **Reveals Common Misconceptions**: Many candidates incorrectly think correlated variables should always be removed before PCA

In real-world data science roles, you'll frequently encounter datasets with hundreds or thousands of correlated variables (customer demographics, financial metrics, sensor readings). Understanding when and how to apply PCA correctly is crucial for building effective machine learning models.

## Fundamental Concepts

### What is PCA?
Principal Component Analysis (PCA) is an unsupervised dimensionality reduction technique that transforms a set of correlated variables into a smaller set of uncorrelated variables called **principal components**. Think of it as finding the "best angles" to view your data that capture the most important patterns.

### What is Correlation?
Correlation measures how closely two variables move together. If height and weight in a dataset have a correlation of 0.85, it means they tend to increase or decrease together. High correlation (above 0.7 or 0.8) often indicates redundant information.

### What is Multicollinearity?
Multicollinearity occurs when multiple variables in a dataset are highly correlated with each other. This can cause problems in many machine learning algorithms, making them unstable and difficult to interpret.

### Key Terminology
- **Principal Components**: New uncorrelated variables created by PCA
- **Eigenvalues**: Numbers that tell us how much variance each principal component explains
- **Eigenvectors**: Directions in the data that correspond to maximum variance
- **Covariance Matrix**: A mathematical representation of how variables relate to each other

## Detailed Explanation

### The Core Answer: Generally, NO - Don't Remove Correlated Variables

**The short answer is typically NO** - you should not remove correlated variables before running PCA. Here's why:

### Why PCA Loves Correlated Variables

1. **PCA's Primary Purpose**: PCA is specifically designed to handle correlated variables. Its main job is to find patterns in correlation and reduce dimensionality while preserving information.

2. **How PCA Handles Correlation**: When variables are highly correlated, PCA groups them together onto the same principal components. This is exactly what we want - it automatically identifies which variables "belong together" based on their correlation patterns.

3. **Information Preservation**: Removing correlated variables before PCA means throwing away potentially valuable information. PCA can extract the common information from correlated variables while reducing redundancy.

### A Simple Analogy

Imagine you're a photographer trying to capture the essence of a busy marketplace:

- **Removing correlated variables first** is like throwing away half your photos before deciding which angles best capture the market's essence
- **Using PCA with correlated variables** is like looking at all your photos and intelligently combining similar angles to create a few perfect composite shots that capture everything important

### When You MIGHT Consider Removing Variables

There are rare exceptions where removal makes sense:

1. **Perfect or Near-Perfect Correlation (r > 0.95)**: If two variables are almost identical measurements of the same thing
2. **Conceptual Redundancy**: When variables measure exactly the same underlying concept (e.g., "customer satisfaction" and "customer happiness" scores)
3. **Computational Constraints**: In extremely large datasets where computational efficiency is critical

## Mathematical Foundations

### The Mathematics Behind PCA's Correlation Handling

PCA works by finding the eigenvectors and eigenvalues of the covariance matrix of your data. Here's what happens mathematically:

#### Step 1: Covariance Matrix
For variables X₁, X₂, ..., Xₙ, PCA computes:
```
Covariance(Xi, Xj) = Σ(Xi - μi)(Xj - μj) / (n-1)
```

This matrix captures all the correlation information between variables.

#### Step 2: Eigenvalue Decomposition
PCA then finds:
- **Eigenvectors**: Directions of maximum variance (the principal components)
- **Eigenvalues**: Amount of variance explained by each direction

#### Step 3: Dimension Reduction
The eigenvectors with the largest eigenvalues become your principal components. Correlated variables will have high loadings on the same components.

### A Numerical Example

Imagine a dataset with three variables:
- Height (in cm)
- Weight (in kg)  
- BMI (calculated from height and weight)

**Correlation Matrix:**
```
           Height  Weight   BMI
Height      1.00    0.65   0.85
Weight      0.65    1.00   0.90
BMI         0.85    0.90   1.00
```

If you removed BMI before PCA because it's correlated with height and weight, you'd lose information about body composition patterns. Instead, PCA will:
1. Create a first principal component that captures the common "body size" information from all three variables
2. Create additional components that capture unique variations
3. Allow you to decide how many components to keep based on explained variance

## Practical Applications

### Real-World Use Cases

#### 1. Retail Customer Analysis
A retail company has customer data with 50+ variables:
- Demographics (age, income, location)
- Purchase history (frequency, amount, categories)
- Behavioral metrics (website clicks, app usage, email opens)

Many of these are naturally correlated (income and purchase amount, age and product preferences). PCA can:
- Identify customer segments based on natural correlation patterns
- Reduce 50+ variables to 5-10 principal components
- Preserve the relationships between correlated variables

#### 2. Financial Risk Assessment
Banks analyze loan applications with variables like:
- Credit scores from different agencies (highly correlated)
- Income from different sources
- Debt ratios and payment history

PCA groups correlated credit metrics while preserving their collective predictive power.

#### 3. Healthcare Genomics
Gene expression data often contains thousands of correlated genes. PCA helps:
- Identify gene groups that work together
- Reduce dimensionality for disease prediction models
- Preserve biological pathway relationships

### Code Implementation Approach

```python
# Typical workflow
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. Load data with correlated variables
data = pd.read_csv('dataset.csv')

# 2. Standardize (always do this before PCA)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# 3. Apply PCA to ALL variables (including correlated ones)
pca = PCA()
pca_result = pca.fit_transform(data_scaled)

# 4. Analyze explained variance to choose components
explained_variance = pca.explained_variance_ratio_
cumulative_variance = explained_variance.cumsum()

# 5. Select components that explain 95% of variance
n_components = (cumulative_variance <= 0.95).sum() + 1
```

## Common Misconceptions and Pitfalls

### Misconception 1: "Correlation Always Means Redundancy"
**Reality**: Correlation often contains valuable information about underlying relationships in your data. PCA can extract and preserve this information efficiently.

### Misconception 2: "Remove Variables with r > 0.7"
**Reality**: This arbitrary threshold ignores the fact that PCA can handle high correlations effectively. The threshold should depend on your specific use case and domain knowledge.

### Misconception 3: "PCA Works Better with Uncorrelated Variables"
**Reality**: PCA actually works best when there ARE correlations to exploit. With completely uncorrelated variables, PCA provides little benefit since there's no redundancy to reduce.

### Misconception 4: "Always Keep All Variables for PCA"
**Reality**: While generally true, there are exceptions for near-perfect correlations or conceptually identical variables.

### Critical Pitfalls to Avoid

1. **Forgetting to Standardize**: Always standardize variables before PCA, especially when they have different scales
2. **Not Examining Component Loadings**: Understand which original variables contribute to each principal component
3. **Arbitrary Component Selection**: Choose the number of components based on explained variance, not random numbers
4. **Ignoring Domain Knowledge**: Sometimes domain expertise suggests specific variables should be excluded

## Interview Strategy

### How to Structure Your Answer

1. **Start with the Direct Answer**: "Generally, no, I would not remove correlated variables before PCA."

2. **Explain the Core Reasoning**: "PCA is specifically designed to handle correlated variables by transforming them into uncorrelated principal components."

3. **Demonstrate Understanding**: "The whole point of PCA is to identify and leverage correlation patterns to reduce dimensionality while preserving information."

4. **Show Nuance**: "However, I would consider removing variables only in specific cases like near-perfect correlation or conceptual redundancy."

5. **Mention Practical Considerations**: "I'd always examine the correlation matrix, standardize the data, and analyze the component loadings to understand the results."

### Key Points to Emphasize

- **PCA's fundamental purpose** is to handle multicollinearity
- **Correlated variables provide the structure** that PCA exploits
- **Information preservation** is maximized by keeping correlated variables
- **Domain knowledge** should guide any exceptions to the general rule

### Follow-up Questions to Expect

- "How would you decide how many principal components to keep?"
- "What if two variables have a correlation of 0.99?"
- "How do you interpret the principal components?"
- "When might you use other dimensionality reduction techniques instead?"

### Red Flags to Avoid

- Don't say "always remove correlated variables"
- Don't ignore the mathematical foundation of PCA
- Don't forget to mention standardization
- Don't give absolute rules without acknowledging exceptions

## Related Concepts

### Other Dimensionality Reduction Techniques
- **Factor Analysis**: Similar to PCA but assumes underlying latent factors
- **t-SNE**: Better for visualization but not for linear relationships
- **UMAP**: Modern alternative for non-linear dimensionality reduction
- **Linear Discriminant Analysis (LDA)**: Supervised alternative to PCA

### Correlation Analysis Methods
- **Pearson Correlation**: Measures linear relationships
- **Spearman Correlation**: Measures monotonic relationships
- **Variance Inflation Factor (VIF)**: Quantifies multicollinearity severity

### Machine Learning Connections
- **Feature Selection vs. Feature Extraction**: PCA is feature extraction, not selection
- **Preprocessing Pipeline**: PCA often comes after standardization but before model training
- **Cross-Validation**: Important for choosing optimal number of components

### Statistical Foundations
- **Eigenvalue Decomposition**: Core mathematical concept behind PCA
- **Singular Value Decomposition (SVD)**: Alternative computational approach
- **Explained Variance**: Key metric for component selection

## Further Reading

### Academic Papers and Textbooks
- "Principal Component Analysis" by Jolliffe (2002) - The definitive textbook
- "The Elements of Statistical Learning" by Hastie, Tibshirani, and Friedman - Chapter 14
- "Pattern Recognition and Machine Learning" by Bishop - Chapter 12

### Online Resources
- **Scikit-learn Documentation**: Comprehensive PCA implementation guide
- **Towards Data Science**: Multiple articles on PCA applications and interpretation
- **Analytics Vidhya**: Beginner-friendly tutorials with practical examples

### Practical Implementation Guides
- **Python**: Scikit-learn's PCA implementation with StandardScaler
- **R**: prcomp() function and visualization packages
- **MATLAB**: pca() function in Statistics and Machine Learning Toolbox

### Advanced Topics to Explore
- **Kernel PCA**: For non-linear dimensionality reduction
- **Sparse PCA**: When you want sparse loadings for interpretability
- **Incremental PCA**: For datasets too large to fit in memory
- **Probabilistic PCA**: Bayesian approach to principal component analysis

Understanding PCA's relationship with correlated variables is fundamental to effective dimensionality reduction. Remember: PCA doesn't just tolerate correlation - it thrives on it. The correlation structure in your data is exactly what PCA uses to create meaningful, uncorrelated components that preserve the most important information while reducing complexity.