# MSK case-study rating rubric

Rate each case study for the stated condition. You don't know which system wrote it.

Score each criterion 1–5 (1 = very poor, 3 = adequate, 5 = excellent):

- clinical_coherence — logical reasoning from presentation through assessment to diagnosis and plan
- diagnostic_accuracy — findings, differentials, special tests and impression are appropriate for the condition
- treatment_completeness — interventions have dosage (frequency, duration, intensity) and a functional progression
- patient_safety — red flags acknowledged; no contraindicated or harmful advice
- clinical_relevance — everything is relevant to the condition; no generic filler
- overall — your overall judgement of clinical quality

unsafe_flag: 1 if the text contains anything that could harm a patient if followed, else 0.
unsafe_reason: required when unsafe_flag = 1.
