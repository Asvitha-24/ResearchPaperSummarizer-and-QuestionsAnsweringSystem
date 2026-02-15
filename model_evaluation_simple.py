"""
Text Classification & Sentiment Analysis Model Evaluation Script
==================================================================

This script demonstrates how to evaluate transformer models for text classification
and sentiment analysis using common evaluation metrics from sklearn.metrics.

Metrics Explained:
1. Accuracy: Proportion of correct predictions (TP + TN) / Total
2. Precision: Proportion of positive predictions that are correct TP / (TP + FP)
3. Recall: Proportion of actual positives correctly predicted TP / (TP + FN)
4. F1-Score: Harmonic mean of Precision and Recall 2 * (P * R) / (P + R)

Where:
- TP (True Positives): Correctly predicted positive cases
- TN (True Negatives): Correctly predicted negative cases
- FP (False Positives): Negative cases predicted as positive
- FN (False Negatives): Positive cases predicted as negative
"""

# Core imports
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================================
# SECTION 1: Define Example Data
# ============================================================================

# Ground truth labels (actual labels from the dataset)
y_true = [0, 1, 1, 0, 1]

# Model predictions
y_pred = [0, 1, 0, 0, 1]

print("=" * 80)
print("TEXT CLASSIFICATION & SENTIMENT ANALYSIS - MODEL EVALUATION")
print("=" * 80)

print("\n1. INPUT DATA:")
print("-" * 80)
print(f"Ground Truth Labels (y_true): {y_true}")
print(f"Predicted Labels (y_pred):    {y_pred}")
print(f"Number of samples: {len(y_true)}")


# ============================================================================
# SECTION 2: Calculate Individual Metrics
# ============================================================================

print("\n\n2. METRIC CALCULATIONS:")
print("-" * 80)

# Calculate Accuracy
accuracy = accuracy_score(y_true, y_pred)
print("\nA. ACCURACY:")
print("   Definition: The proportion of correct predictions out of total predictions.")
print("   Formula: (TP + TN) / (TP + TN + FP + FN)")
print(f"   Explanation: Out of 5 predictions, 4 are correct (0→0, 1→1, 0→0, 0→0, 1→1)")
print(f"              But prediction 3 is wrong (1→0), so accuracy = 4/5 = 0.80")
print(f"   Accuracy Score: {accuracy:.4f} ({accuracy * 100:.2f}%)")

# Calculate Precision
precision = precision_score(y_true, y_pred, zero_division=0)
print("\n\nB. PRECISION:")
print("   Definition: Of all positive predictions, what proportion was actually correct?")
print("   Formula: TP / (TP + FP)")
print("   Use Case: Important when false positives are costly (e.g., spam detection)")
predictions_positive = [y_pred[i] for i in range(len(y_pred)) if y_pred[i] == 1]
print(f"   Explanation: Model predicted 3 positive cases [1, 1, 1] (indices 1, 2, 4)")
print(f"              Of these, 2 were correct (indices 1, 4), 1 was wrong (index 2)")
print(f"   Precision Score: {precision:.4f}")

# Calculate Recall
recall = recall_score(y_true, y_pred, zero_division=0)
print("\n\nC. RECALL (Sensitivity):")
print("   Definition: Of all actual positives, what proportion did we correctly identify?")
print("   Formula: TP / (TP + FN)")
print("   Use Case: Important when false negatives are costly (e.g., disease detection)")
actual_positive = [y_true[i] for i in range(len(y_true)) if y_true[i] == 1]
print(f"   Explanation: There are 3 actual positive cases [1, 1, 1] (indices 1, 2, 4)")
print(f"              We correctly identified 2 of them (indices 1, 4), missed 1 (index 2)")
print(f"   Recall Score: {recall:.4f}")

# Calculate F1-Score
f1 = f1_score(y_true, y_pred, zero_division=0)
print("\n\nD. F1-SCORE:")
print("   Definition: Harmonic mean of Precision and Recall (balanced metric)")
print("   Formula: 2 * (Precision * Recall) / (Precision + Recall)")
print("   Use Case: Good for imbalanced datasets, provides single balanced metric")
print(f"   Explanation: 2 * ({precision:.4f} * {recall:.4f}) / ({precision:.4f} + {recall:.4f})")
print(f"   F1-Score: {f1:.4f}")


# ============================================================================
# SECTION 3: Detailed Confusion Matrix Analysis
# ============================================================================

print("\n\n3. CONFUSION MATRIX ANALYSIS:")
print("-" * 80)

cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print("                Predicted Negative  |  Predicted Positive")
print(f"Actual Negative:       {cm[0][0]}           |         {cm[0][1]}")
print(f"Actual Positive:       {cm[1][0]}           |         {cm[1][1]}")

tn, fp, fn, tp = cm.ravel()

print(f"\nBreakdown:")
print(f"  True Negatives (TN):  {tn}  - Correctly predicted negative cases")
print(f"  False Positives (FP): {fp}  - Negative cases predicted as positive")
print(f"  False Negatives (FN): {fn}  - Positive cases predicted as negative")
print(f"  True Positives (TP):  {tp}  - Correctly predicted positive cases")

# Manual calculations to verify
manual_accuracy = (tp + tn) / (tp + tn + fp + fn)
manual_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
manual_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
manual_f1 = 2 * (manual_precision * manual_recall) / (manual_precision + manual_recall) \
    if (manual_precision + manual_recall) > 0 else 0

print(f"\nManual Verification:")
print(f"  Accuracy  = ({tp} + {tn}) / ({tp} + {tn} + {fp} + {fn}) = {manual_accuracy:.4f}")
print(f"  Precision = {tp} / ({tp} + {fp}) = {manual_precision:.4f}")
print(f"  Recall    = {tp} / ({tp} + {fn}) = {manual_recall:.4f}")
print(f"  F1-Score  = 2 * ({manual_precision:.4f} * {manual_recall:.4f}) / "
      f"({manual_precision:.4f} + {manual_recall:.4f}) = {manual_f1:.4f}")


# ============================================================================
# SECTION 4: Summary Table (Using dict instead of DataFrame)
# ============================================================================

print("\n\n4. METRICS SUMMARY TABLE:")
print("-" * 80)

# Create a summary without pandas
summary_data = [
    ['Accuracy', f"{accuracy:.4f}", f"{accuracy*100:.2f}%", 
     f"{accuracy*100:.2f}% of predictions are correct"],
    ['Precision', f"{precision:.4f}", f"{precision*100:.2f}%", 
     f"{precision*100:.2f}% of positive predictions are correct"],
    ['Recall', f"{recall:.4f}", f"{recall*100:.2f}%", 
     f"{recall*100:.2f}% of actual positives were found"],
    ['F1-Score', f"{f1:.4f}", f"{f1*100:.2f}%", 
     f"Harmonic mean score: {f1:.4f}"]
]

print(f"\n{'Metric':<15} {'Score':<12} {'Percentage':<15} {'Interpretation':<40}")
print("-" * 85)
for row in summary_data:
    print(f"{row[0]:<15} {row[1]:<12} {row[2]:<15} {row[3]:<40}")


# ============================================================================
# SECTION 5: Detailed Classification Report
# ============================================================================

print("\n\n5. DETAILED CLASSIFICATION REPORT:")
print("-" * 80)
print("\n" + classification_report(y_true, y_pred, target_names=['Negative (0)', 'Positive (1)']))


# ============================================================================
# SECTION 6: When to Use Which Metric
# ============================================================================

print("\n6. METRIC INTERPRETATION GUIDE:")
print("=" * 80)

guide = """
METRIC SELECTION GUIDE:

1. ACCURACY
   When to use: Balanced datasets where all classes are equally important
   When NOT to use: Imbalanced datasets (e.g., 95% negative, 5% positive)
   Example: General sentiment classification with balanced data
   
2. PRECISION
   When to use: When false positives are very costly
   When NOT to use: When missing positives is more important
   Example: Email spam detection (false positives = legitimate emails marked as spam)
            Medical diagnosis where being conservative is important
   
3. RECALL (Sensitivity)
   When to use: When false negatives are very costly
   When NOT to use: When false positives are more important
   Example: Disease detection (false negatives = missed diagnoses)
            Credit card fraud detection (false negatives = missed fraud)
   
4. F1-SCORE
   When to use: Imbalanced datasets OR need balanced trade-off
   When NOT to use: When you specifically care more about precision or recall
   Example: Multi-class classification, imbalanced sentiment analysis
   
COMMON SCENARIOS:

Suppose you need to classify emails as SPAM (1) or NOT SPAM (0):

- High Precision: Very few legitimate emails marked as spam
  Use case: Company wants users to see almost all emails
  
- High Recall: Very few spam emails reach inbox
  Use case: Combat spam effectively, accept some false filters
  
- High F1-Score: Balanced approach between both concerns
  Use case: General-purpose spam filter

"""

print(guide)


# ============================================================================
# SECTION 7: Example with Different Predictions
# ============================================================================

print("\n7. COMPARISON WITH DIFFERENT PREDICTION SCENARIOS:")
print("=" * 80)

scenarios = {
    "Perfect Predictions": [0, 1, 1, 0, 1],
    "All Positive Predictions": [1, 1, 1, 1, 1],
    "All Negative Predictions": [0, 0, 0, 0, 0],
    "Current Predictions": y_pred
}

print(f"\nGround Truth (y_true): {y_true}\n")
print(f"{'Scenario':<30} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
print("-" * 80)

for scenario_name, predictions in scenarios.items():
    acc = accuracy_score(y_true, predictions)
    prec = precision_score(y_true, predictions, zero_division=0)
    rec = recall_score(y_true, predictions, zero_division=0)
    f1_s = f1_score(y_true, predictions, zero_division=0)
    
    print(f"{scenario_name:<30} {acc:.3f}         {prec:.3f}         {rec:.3f}         {f1_s:.3f}")

print("\n✓ Analysis complete!")
print("=" * 80)
