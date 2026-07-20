"""Model card and limitations document for LLM system context.

# LESSON POINT: Model Cards as Grounding Context
# In an LLMOps setup, the language model needs structured context to produce
# grounded, safe responses. A model card describes the supervised model's
# capabilities, limitations, and known biases. Injecting this as system context
# prevents the LLM from making unsupported claims (e.g., causal explanations)
# and ensures it communicates limitations to the user.
"""

MODEL_CARD = """
## Model Card: Titanic Survival Classifier

### Overview
- Model type: Logistic Regression with preprocessing pipeline
- Framework: scikit-learn
- Task: Binary classification (survived / did not survive)
- Dataset: Titanic passenger manifest (historical, 891 base records)

### Features used
- Pclass (ticket class: 1, 2, or 3)
- Sex (male or female)
- Age (years, may be missing)
- SibSp (number of siblings or spouses aboard)
- Parch (number of parents or children aboard)
- Fare (ticket price)
- Embarked (port of embarkation: S, C, or Q)

### Known limitations
1. This model is trained on historical data from 1912. It reflects the social
   conditions of that era (e.g., "women and children first" evacuation policy).
2. The model is a logistic regression. It captures linear relationships between
   features and survival but cannot model complex interactions.
3. Feature importance does not imply causation. For example, the model may show
   that being female increases predicted survival probability, but this reflects
   a historical evacuation pattern, not a causal mechanism.
4. Missing age values are imputed with the median. Predictions for passengers
   with missing age are less reliable.
5. The model has not been validated on data outside the Titanic dataset. It
   should not be used for any real-world decision-making.

### Intended use
This model is a demonstration of MLOps practices: training, versioning, serving,
monitoring, and retraining. It is not a production decision system.

### Ethical considerations
The dataset reflects historical biases in survival outcomes related to gender,
class, and age. The model reproduces these biases by design (it predicts based
on historical patterns). Users should not interpret model outputs as endorsing
or justifying those patterns.
""".strip()
