# Multi-Label Text Categorization using ML

A news headline can belong to **multiple categories** at once.
Example: `"India wins cricket World Cup"` → **[Sports, India]**

## Categories (8 total)
`Politics` · `Sports` · `Technology` · `Business` · `Health` · `Entertainment` · `Science` · `India`

## ML Pipeline

```
Raw Text → Preprocessing → TF-IDF Vectorization → OneVsRest Classifier → Multi Labels
```

## Models Used

| Model | Strategy |
|-------|----------|
| Logistic Regression | OneVsRest — one binary classifier per label |
| Linear SVM | OneVsRest — fast & accurate for text |
| Random Forest | OneVsRest — ensemble approach |

## Evaluation Metrics

| Metric | Meaning |
|--------|---------|
| **Hamming Loss** | Lower is better — % of wrong labels |
| **F1-Micro** | Overall F1 score across all labels |
| **F1-Macro** | Average F1 per category |
| **Subset Accuracy** | All labels must match exactly |

## How to Run

### Step 1 — Install
```bash
pip install -r requirements.txt
```

### Step 2 — Run
```bash
python multilabel_categorization.py
```

### Output
- Terminal lo model comparison + predictions print avutayi
- `results.json` auto-save avutundi

## File Structure

```
multilabel_text/
├── multilabel_categorization.py   ← Main file
├── requirements.txt
├── README.md
└── results.json                   ← Generated after run
```

## Key Concepts

- **Multi-Label** vs Multi-Class: one sample can have MANY labels
- **TF-IDF**: converts words to numbers based on frequency & importance
- **OneVsRest**: trains one binary classifier per category
- **Hamming Loss**: fraction of incorrectly predicted labels
