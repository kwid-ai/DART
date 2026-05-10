# DART: Degradation-Aware Post-Distillation Refinement with Teacher-Agreement Rewards 

Large language models can generate clinically rich physiotherapy case studies, but their computational cost limits deployment in resource-constrained clinical and educational environments. 
We present a framework that diagnoses post-distillation layer degradation using teacher-student KL divergence, uses that signal to both select layers for LoRA adaptation and allocate rank capacity to them, and keeps the teacher active as a reward anchor during refinement.

