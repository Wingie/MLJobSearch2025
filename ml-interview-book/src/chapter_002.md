# Train-Test Split Ratios: Beyond the 80:20 Rule

## The Interview Question
> **Meta/Google/OpenAI**: "Is it always necessary to use an 80:20 ratio for the train test split? If not, how would you decide on a split?"

## Why This Question Matters

This question is a favorite among top tech companies because it tests multiple critical skills that separate junior from senior machine learning practitioners:

- **Statistical Understanding**: Do you understand the mathematical trade-offs between training and testing data?
- **Practical Experience**: Have you worked with real datasets where the standard ratio doesn't apply?
- **Domain Knowledge**: Can you adapt your approach based on specific use cases and constraints?
- **Resource Awareness**: Do you consider computational and time constraints in your decisions?

Companies like Meta, Google, and OpenAI ask this because they deal with diverse datasets—from massive user interaction logs to specialized research datasets—where blindly applying 80:20 can lead to suboptimal models or wasted resources.

## Fundamental Concepts

### What is Train-Test Split?

Train-test split is the practice of dividing your dataset into separate portions:
- **Training Set**: Data used to teach the model patterns and relationships
- **Test Set**: Data held back to evaluate how well the model performs on unseen data

Think of it like studying for an exam. Your textbook and practice problems are your "training data"—you learn from them. The actual exam questions are your "test data"—they evaluate whether you truly understand the material or just memorized specific examples.

### Why Split Data at All?

The main purpose is to detect overfitting—when a model memorizes training examples rather than learning generalizable patterns. Without a separate test set, you'd be like a student grading their own homework using the answer key. The results would look perfect but wouldn't reflect real understanding.

### The Three-Way Split

Many real-world applications use a three-way split:
- **Training Set (60-80%)**: Teaches the model
- **Validation Set (10-20%)**: Tunes hyperparameters and model selection
- **Test Set (10-20%)**: Final, unbiased performance evaluation

## Detailed Explanation

### The 80:20 "Rule" and Its Origins

The 80:20 split became popular because it offers a practical balance:
- **80% for training**: Provides enough data for most algorithms to learn effectively
- **20% for testing**: Gives sufficient samples for reliable performance estimates

However, this ratio emerged from early machine learning practice when datasets were typically small (thousands to tens of thousands of samples). It's not a mathematical law but rather a rule of thumb that worked well in common scenarios.

### When 80:20 Makes Sense

**Medium-sized datasets (1,000-100,000 samples)**: The classic 80:20 split works well because:
- Training set is large enough for learning
- Test set provides adequate statistical power
- Computational resources aren't a major constraint

**Balanced datasets**: When all classes are well-represented, random 80:20 splits maintain class distributions in both sets.

**Standard algorithms**: Traditional machine learning algorithms (logistic regression, random forests, SVMs) often perform well with this ratio.

### When to Deviate from 80:20

#### Small Datasets (< 1,000 samples)
**Problem**: Not enough data for reliable train-test split
**Solution**: Use cross-validation instead

With only 100 samples, an 80:20 split gives you 80 training examples and 20 test examples. This test set is too small for reliable performance estimates, and you're wasting 20% of precious training data.

**Alternative**: 5-fold or 10-fold cross-validation uses all data for both training and testing across multiple iterations.

#### Large Datasets (> 1,000,000 samples)
**Problem**: 20% of a million samples (200,000) for testing is overkill
**Solution**: Use smaller test percentages like 99:1 or 95:5

Example: With 10 million samples, even a 1% test set gives you 100,000 test examples—more than enough for reliable evaluation. This leaves 99% (9.9 million) for training, potentially improving model performance.

#### Imbalanced Datasets
**Problem**: Random splits might not preserve class distributions
**Example**: Disease detection dataset with 95% healthy, 5% diseased patients

**Solution**: Use stratified sampling to maintain class proportions in all splits.

#### Time Series Data
**Problem**: Future data can't predict past events
**Solution**: Use chronological splits, not random splits

For stock price prediction, you might use:
- Training: 2020-2022 data
- Validation: 2023 Q1-Q3 data  
- Test: 2023 Q4 data

#### Computational Constraints
**Problem**: Limited time or computing resources
**Solutions**:
- Use smaller training sets if model training is the bottleneck
- Use smaller test sets if evaluation is expensive
- Consider the cost of data collection vs. model improvement

### Domain-Specific Considerations

#### Medical Applications
- Smaller test sets acceptable due to high cost of data collection
- Focus on ensuring test set represents real clinical conditions
- May use 90:10 or even 95:5 splits

#### Computer Vision
- Large datasets common (ImageNet has millions of images)
- Often use 98:1:1 (train:validation:test) ratios
- Evaluation can be computationally expensive

#### Natural Language Processing
- Dataset size varies enormously
- Small specialized datasets might use cross-validation
- Large pre-training datasets use minimal test percentages

## Mathematical Foundations

### Statistical Power and Sample Size

The reliability of your test set depends on its size. For classification accuracy, the standard error is approximately:

```
Standard Error ≈ √(accuracy × (1 - accuracy) / n)
```

Where `n` is the test set size.

**Example**: With 90% accuracy and 1,000 test samples:
```
Standard Error ≈ √(0.9 × 0.1 / 1000) ≈ 0.0095 ≈ 1%
```

This means your accuracy estimate is 90% ± 2% (roughly 2 standard errors) with 95% confidence.

**Key Insight**: Doubling test set size only reduces uncertainty by √2 ≈ 1.4x. Going from 1,000 to 10,000 test samples improves precision from ±2% to ±0.6%—helpful but with diminishing returns.

### Training Set Size vs. Performance

Most algorithms follow a learning curve where performance improves with more training data but with diminishing returns. The relationship often follows:

```
Performance ≈ a - b × e^(-c × training_size)
```

This means:
- Initial data is very valuable
- Additional data helps but with decreasing benefit
- Eventually, you hit a plateau where more data doesn't help

### Bias-Variance Trade-off in Splits

- **Larger training sets**: Reduce variance (model is more stable) but might increase bias if test set becomes too small for reliable evaluation
- **Larger test sets**: Reduce evaluation variance but might increase model variance due to insufficient training data

## Practical Applications

### Real-World Split Strategies

#### Google Search (Hypothetical)
- **Dataset**: Billions of queries
- **Split**: 99.9:0.05:0.05 (train:validation:test)
- **Reasoning**: 0.05% of billions is still millions of test examples

#### Medical Diagnosis
- **Dataset**: 10,000 patient records
- **Split**: 70:15:15 or 80:10:10
- **Reasoning**: Need sufficient test data for regulatory approval, but data collection is expensive

#### Startup with Limited Data
- **Dataset**: 500 examples
- **Split**: Use 5-fold cross-validation
- **Reasoning**: Every sample is precious; can't afford to "waste" 20% on testing

### Implementation Guidelines

#### Step 1: Assess Your Dataset
```python
# Pseudocode for decision making
if dataset_size < 1000:
    use_cross_validation()
elif dataset_size > 1000000:
    consider_smaller_test_percentage()
elif has_time_component():
    use_temporal_split()
elif is_imbalanced():
    use_stratified_split()
else:
    default_80_20_split()
```

#### Step 2: Consider Your Constraints
- **Time constraints**: Smaller training sets train faster
- **Computational limits**: Smaller test sets evaluate faster
- **Accuracy requirements**: Higher stakes need larger test sets
- **Data cost**: Expensive data collection favors larger training sets

#### Step 3: Validate Your Choice
- Check that your test set size gives adequate statistical power
- Ensure training set is large enough for your algorithm
- Verify that split preserves important data characteristics

## Common Misconceptions and Pitfalls

### Misconception 1: "80:20 is Always Optimal"
**Reality**: The optimal split depends on dataset size, domain, and constraints. Netflix doesn't use the same ratio as a medical researcher with 100 patient samples.

### Misconception 2: "Bigger Test Sets Always Give Better Estimates"
**Reality**: Beyond a certain point, making test sets larger provides diminishing returns while potentially hurting model performance due to reduced training data.

### Misconception 3: "Random Splits Always Work"
**Reality**: Time series data, grouped data (multiple samples from same patients), and hierarchical data require specialized splitting strategies.

### Misconception 4: "Cross-Validation Can Always Replace Train-Test Split"
**Reality**: Cross-validation is great for small datasets but can be computationally prohibitive for very large datasets or complex models.

### Common Pitfalls

#### Data Leakage in Splitting
**Problem**: Related samples end up in both training and test sets
**Example**: Using different photos of the same person in both sets for face recognition
**Solution**: Split by person, not by photo

#### Temporal Leakage
**Problem**: Using future information to predict past events
**Example**: Predicting stock prices using data that wouldn't have been available at prediction time
**Solution**: Strict chronological splits with no overlap

#### Evaluation Set Reuse
**Problem**: Repeatedly evaluating on the same test set and adjusting based on results
**Effect**: Test performance becomes overly optimistic
**Solution**: Use separate validation set for model development, reserve test set for final evaluation only

## Interview Strategy

### Structure Your Answer

1. **Acknowledge the trade-off**: "The 80:20 split balances training data quantity with evaluation reliability, but it's not universally optimal."

2. **Discuss key factors**:
   - Dataset size
   - Domain requirements
   - Computational constraints
   - Data characteristics (time series, imbalanced, etc.)

3. **Provide specific examples**: Show you understand when to deviate

4. **Mention alternatives**: Cross-validation, stratified sampling, temporal splits

### Key Points to Emphasize

- **No universal rule**: The optimal split depends on context
- **Statistical considerations**: Test set size affects reliability of performance estimates
- **Practical constraints**: Time, compute, and data costs matter
- **Domain expertise**: Different fields have different standards and requirements

### Sample Answer Framework

"No, 80:20 isn't always necessary. The optimal split depends on several factors:

For dataset size: With small datasets under 1,000 samples, I'd use cross-validation instead. With very large datasets over a million samples, I might use 95:5 or even 99:1 since 1% of a million is still 10,000 test samples.

For domain considerations: Time series data requires chronological splits, not random ones. Medical data might use smaller test sets due to collection costs, while computer vision with abundant data can afford larger test sets.

For computational constraints: If training time is the bottleneck, I might use less training data. If evaluation is expensive, I might use a smaller test set.

The key is ensuring your test set is large enough for reliable estimates while giving your model sufficient training data to learn effectively."

### Follow-up Questions to Expect

- "How would you handle imbalanced datasets when splitting?"
- "What's the difference between validation and test sets?"
- "When would you use cross-validation instead of a simple split?"
- "How do you split time series data?"
- "What's the minimum size for a reliable test set?"

### Red Flags to Avoid

- Saying 80:20 is always correct
- Not mentioning cross-validation for small datasets
- Ignoring domain-specific considerations
- Not discussing the statistical basis for test set sizing
- Forgetting about data leakage issues

## Related Concepts

### Cross-Validation Techniques
- **K-fold**: Divides data into k equal parts, trains on k-1, tests on 1
- **Stratified**: Maintains class proportions across folds
- **Time series**: Respects temporal order (TimeSeriesSplit)
- **Group**: Ensures related samples stay together

### Model Selection vs. Evaluation
- **Validation set**: Used during development for hyperparameter tuning
- **Test set**: Used only once for final, unbiased evaluation
- **Cross-validation**: Can serve both purposes depending on implementation

### Sampling Strategies
- **Random sampling**: Works for i.i.d. data
- **Stratified sampling**: Preserves class distributions
- **Systematic sampling**: Takes every nth sample
- **Cluster sampling**: Samples groups rather than individuals

### Bias and Variance in Model Evaluation
- **Selection bias**: Non-representative test sets
- **Evaluation variance**: Unstable estimates due to small test sets
- **Overfitting to test set**: Implicit optimization based on test performance

## Further Reading

### Academic Papers
- "Why 70/30 or 80/20 Relation Between Training and Testing Sets: A Pedagogical Explanation" by Gholamy et al. (2018)
- "On Splitting Training and Validation Set: A Comparative Study" by Jiang et al. (2018)
- "Cross-validation procedures for model selection" by Arlot & Celisse (2010)

### Practical Guides
- Scikit-learn documentation on cross-validation
- "Hands-On Machine Learning" by Aurélien Géron (Chapter 2: End-to-End Machine Learning Project)
- "The Elements of Statistical Learning" by Hastie et al. (Chapter 7: Model Assessment and Selection)

### Online Resources
- Fast.ai practical deep learning course (data splitting best practices)
- Google's Machine Learning Crash Course (validation and test sets)
- Andrew Ng's Machine Learning Course (bias-variance trade-off)

### Industry Examples
- Netflix Prize competition (evaluation methodology)
- ImageNet competition (large-scale dataset splitting)
- Kaggle competitions (various splitting strategies in practice)