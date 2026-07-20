"""Data drift simulation - inject synthetic passengers with inverted patterns."""

import logging

import numpy as np
import pandas as pd

from .config import settings

logger = logging.getLogger(__name__)


# LESSON POINT: Simulating Data Drift in Development
# In production, data drift happens naturally (e.g., changes in user behavior, seasonal trends,
# or demographic shifts). To test and validate our monitoring dashboard in dev, this function
# deliberately violates the statistical patterns of the original Titanic dataset (e.g., making
# first-class males survive and third-class females die, along with extreme ages and fares).
# Appending this artificial shift lets us check that Evidently AI flags the anomaly correctly.
def simulate_drift(n_samples: int = 100) -> dict:
    """Generate *n_samples* synthetic passengers whose survival patterns
    are deliberately inverted compared to the real Titanic data, then
    append them to the working dataset.

    Typical Titanic patterns:
        - 1st-class females almost always survived
        - 3rd-class males almost always died

    Drifted patterns (inverted):
        - 1st-class males survive
        - 3rd-class females die
        - Extreme age / fare distributions
    """
    rng = np.random.default_rng()

    df = pd.read_csv(settings.DATA_PATH)
    next_id = int(df["PassengerId"].max()) + 1

    rows: list[dict] = []
    for i in range(n_samples):
        sex = rng.choice(["male", "female"])
        pclass = int(rng.choice([1, 2, 3]))

        # Invert survival: 1st-class males survive, 3rd-class females don't
        if sex == "male" and pclass == 1:
            survived = 1
        elif sex == "female" and pclass == 3:
            survived = 0
        else:
            survived = int(rng.choice([0, 1]))

        # Extreme distributions for age and fare
        age = float(rng.choice([rng.uniform(80, 100), rng.uniform(0, 2)]))
        fare = float(rng.uniform(0, 600))

        rows.append(
            {
                "PassengerId": next_id + i,
                "Survived": survived,
                "Pclass": pclass,
                "Name": f"DriftPassenger, Synthetic {next_id + i}",
                "Sex": sex,
                "Age": round(age, 1),
                "SibSp": int(rng.integers(0, 9)),
                "Parch": int(rng.integers(0, 7)),
                "Ticket": f"DRIFT{next_id + i}",
                "Fare": round(fare, 4),
                "Cabin": "",
                "Embarked": rng.choice(["S", "C", "Q"]),
            }
        )

    drift_df = pd.DataFrame(rows)
    drift_df.to_csv(settings.DATA_PATH, mode="a", header=False, index=False)

    total_rows = len(df) + n_samples
    logger.info("Injected %d drifted samples -> dataset now has %d rows", n_samples, total_rows)

    return {
        "n_injected": n_samples,
        "total_rows": total_rows,
        "message": f"Injected {n_samples} drifted samples into dataset",
    }
