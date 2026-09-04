"""Frozen clinical-domain taxonomy shared by templates and evaluation metrics.

Keeping authored domain knowledge outside the scoring modules prevents the deterministic domain
template from depending on evaluation implementation details.
"""

# Resolved taxonomy decisions:
# - FOG is assigned to gait_mobility because it directly describes gait freezing.
# - Mobility Summary is assigned to gait_mobility because it summarizes mobility-related
#   activities; its non-monotonic scale is preserved in the table interpretation.

TAXONOMY_VERSION = "draft-v5"

# FEATURE_CATEGORY defines category membership using stable, machine-readable identifiers.
FEATURE_CATEGORY = {
    "FOG (Freezing of Gait)": "gait_mobility",
    "Gait": "gait_mobility",
    "Mobility Summary": "gait_mobility",
    "Postural Stability": "gait_mobility",
    "Postural Instability at Dx": "gait_mobility",
    "UPDRS-III Total": "motor_impairment",
    "Rigidity at Dx": "motor_impairment",
    "H&Y Stage": "motor_impairment",
    "UPDRS-IV Total": "motor_complications",
    "MoCA": "cognition",
    "GDS-15": "mood_symptoms",
    "Lightheadedness on Standing": "autonomic_symptoms",
    "Fainting": "autonomic_symptoms",
    "Postural Hypotension": "autonomic_symptoms",
    "Urinary Problems": "autonomic_symptoms",
    "UPDRS-I Total": "overall_non_motor_burden",
    "Daytime Sleepiness": "sleep_wake_symptoms",
    "Age": "demographics",
    "BMI": "anthropometric",
    "PD Duration": "disease_course",
    "Dopaminergic Therapy": "treatment_status",
}

CATEGORY_TO_FEATURES = {}
for _feature, _category in FEATURE_CATEGORY.items():
    CATEGORY_TO_FEATURES.setdefault(_category, set()).add(_feature)


# CATEGORY_LABELS defines how each internal category is written in clinician-facing prose.
CATEGORY_LABELS = {
    "gait_mobility": "gait and mobility",
    "motor_impairment": "motor impairment",
    "motor_complications": "motor complications",
    "cognition": "cognition",
    "mood_symptoms": "mood symptoms",
    "autonomic_symptoms": "autonomic symptoms",
    "overall_non_motor_burden": "overall non-motor burden",
    "sleep_wake_symptoms": "sleep/wake symptoms",
    "demographics": "demographic characteristics",
    "anthropometric": "anthropometric characteristics",
    "disease_course": "disease-course characteristics",
    "treatment_status": "treatment status",
}

CATEGORY_PHRASES = {
    "gait_mobility": [
        "gait and mobility", "gait and balance", "gait-related", "gait assessment",
        "gait assessments", "gait characteristic", "gait characteristics", "gait impairment",
        "mobility", "mobility-related",
        "mobility assessment", "mobility assessments", "mobility characteristic",
        "mobility characteristics", "mobility difficulties",
    ],
    "motor_impairment": [
        "motor impairment", "motor examination", "motor severity", "motor burden",
        "motor characteristic", "motor characteristics", "motor assessment", "motor assessments",
        "motor feature", "motor features", "motor performance", "motor function", "motor domain",
        "motor manifestation", "motor manifestations", "motor symptom", "motor symptoms",
        "motor evaluation", "overall motor",
    ],
    "motor_complications": ["motor complications", "dyskinesia", "motor fluctuations"],
    "cognition": ["cognitive factors", "cognitive measures"],
    "mood_symptoms": ["mood symptoms", "mood-related factors"],
    "autonomic_symptoms": [
        "autonomic symptoms", "autonomic features", "autonomic factors",
    ],
    "overall_non_motor_burden": [
        "overall non-motor burden", "non-motor aspects of daily living",
        "non-motor experiences", "non-motor burden",
    ],
    "sleep_wake_symptoms": ["sleep/wake symptoms", "sleep-wake symptoms"],
    "demographics": ["demographic characteristics", "demographic factors"],
    "anthropometric": ["anthropometric characteristics", "anthropometric factors"],
    "disease_course": ["disease-course characteristics", "disease-course factors"],
    "treatment_status": ["treatment status", "treatment characteristics"],
}
