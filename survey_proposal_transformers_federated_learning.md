# Survey Proposal: Transformers and Distributed Attention in Federated Learning

**Author:** Hira Aman, Year 1 PhD Student  
**Course:** PhD Mini-course on Ensuring Trustworthiness in Federated Learning  
**Supervisor:** Professor Fischella  
**Intermediate Presentation:** July 13, 2026 at 15:00  
**Final Deadline:** August 31, 2026  

---

## 1. Survey Topic and Motivation

### Topic
**Transformers and Distributed Attention Mechanisms in Federated Learning: Architectures, Challenges, and Trustworthiness**

### Motivation

Transformer architectures have become the dominant paradigm in modern machine learning, powering large language models (LLMs), vision transformers (ViTs), and multimodal systems. Their core mechanism — self-attention — computes pairwise relationships across all input tokens, making them exceptionally expressive but also computationally and communicatively expensive. This creates a fundamental tension when deploying Transformers in Federated Learning (FL) settings, where data is distributed across heterogeneous clients with limited bandwidth and compute.

Several converging trends make this survey timely:

1. **Scale vs. Communication**: Pre-trained Transformer models (e.g., BERT, GPT, ViT) have billions of parameters. Naively federating their training is prohibitively expensive. Novel approaches — such as partial fine-tuning, adapter layers, and LoRA — are emerging to address this.

2. **Distributed Attention**: The attention mechanism itself can be reformulated in a distributed or privacy-preserving manner, opening new research directions beyond simple gradient aggregation.

3. **Trustworthiness concerns**: Attention weights are not inherently private; they may leak client data distributions. This intersects directly with the course theme of trustworthiness.

4. **Scientific applications**: In domains like Raman spectroscopy and other spectral analysis tasks, Transformers operating on sequential spectral data present a compelling use case for federated deployment across research institutions that cannot share raw measurements.

This survey aims to systematically organize this rapidly growing body of work, identify open problems, and propose a taxonomy of approaches at the intersection of Transformers and FL.

---

## 2. Tentative Table of Contents

```
1. Introduction
   1.1 Background and Scope
   1.2 Why Transformers in FL? Motivation and Challenges
   1.3 Survey Methodology and Paper Selection Criteria
   1.4 Organization of the Survey

2. Preliminaries
   2.1 Federated Learning: Setup, Objectives, and Threat Models
   2.2 Transformer Architectures: Self-Attention, Encoder-Decoder, ViT
   2.3 Standard FL Algorithms: FedAvg, FedProx, SCAFFOLD, FedOpt
   2.4 Trustworthiness Dimensions: Privacy, Fairness, Robustness, Accountability

3. Federated Fine-tuning of Pre-trained Transformers
   3.1 Full Fine-tuning under Communication Constraints
   3.2 Parameter-Efficient Fine-tuning (PEFT): LoRA, Adapters, Prefix Tuning
   3.3 Federated Prompt Tuning
   3.4 Split Learning Variants for Transformers

4. Distributed and Privacy-Preserving Attention Mechanisms
   4.1 Reformulating Self-Attention for Distributed Computation
   4.2 Secure Multi-Party Computation (SMPC) for Attention
   4.3 Differential Privacy Applied to Attention Weights and Gradients
   4.4 Homomorphic Encryption Approaches

5. Communication Efficiency in Federated Transformer Training
   5.1 Gradient Compression and Sparsification
   5.2 Knowledge Distillation-Based FL for Transformers
   5.3 Asynchronous and Partial Participation Strategies
   5.4 Model Heterogeneity: Federated Learning with Different Transformer Sizes

6. Robustness and Trustworthiness
   6.1 Byzantine Robustness in Transformer FL
   6.2 Backdoor and Poisoning Attacks on Federated Transformers
   6.3 Privacy Leakage via Attention Weights and Gradients
   6.4 Fairness and Personalization Across Heterogeneous Clients

7. Applications
   7.1 Natural Language Processing (NLP) and LLMs
   7.2 Vision Transformers in Medical and Scientific Imaging
   7.3 Multimodal Federated Learning
   7.4 Scientific Data: Spectroscopy, Time Series, and Sensor Networks

8. Benchmarks, Datasets, and Evaluation Settings
   8.1 Standard FL Benchmarks
   8.2 Transformer-Specific Evaluation Protocols
   8.3 Trustworthiness Evaluation Metrics
   8.4 Open Frameworks and Toolkits

9. Open Challenges and Future Directions
   9.1 Scaling Laws in Federated Transformer Training
   9.2 Foundation Models and Cross-Silo Federation
   9.3 Theoretical Guarantees for Distributed Attention
   9.4 Towards Truly Trustworthy Federated Transformers

10. Conclusion
```

---

## 3. Section Descriptions

### Section 1 — Introduction
Frames the survey's scope and contributions. Explains why the intersection of Transformers and FL is both practically important and theoretically rich. Defines what is and is not covered (e.g., we focus on Transformer-specific challenges, not general FL surveys). Describes the literature search process (keyword combinations, databases used: arXiv, IEEE Xplore, ACM DL, NeurIPS/ICML/ICLR proceedings).

### Section 2 — Preliminaries
Provides self-contained background. Covers the FL problem formulation (data heterogeneity, communication rounds, aggregation), the Transformer architecture (multi-head self-attention complexity O(n²d)), and the standard FL algorithms that form the baseline. Introduces the trustworthiness framework used throughout the survey.

### Section 3 — Federated Fine-tuning of Pre-trained Transformers
The most practically active area. Surveys how large pre-trained Transformers (BERT, RoBERTa, GPT-2, ViT) are adapted for FL without transmitting full model updates. Covers PEFT methods (LoRA inserts low-rank matrices; adapters insert small bottleneck modules) and analyzes their communication-accuracy tradeoffs. Includes federated prompt tuning where only soft prompts are communicated.

### Section 4 — Distributed and Privacy-Preserving Attention
Explores restructuring the attention computation itself for distributed settings. Covers linearized attention approximations that decompose global attention into aggregatable local statistics, SMPC protocols for dot-product attention, and differentially private attention where noise is injected into attention scores or output projections.

### Section 5 — Communication Efficiency
Surveys techniques to reduce the per-round communication cost, which is severe for Transformers. Covers gradient compression (top-k sparsification, quantization), ensemble/distillation-based approaches (FedDF, DS-FL) where clients only share soft labels, and heterogeneous federation where clients run different-sized Transformers (sub-model extraction, HeteroFL).

### Section 6 — Robustness and Trustworthiness
Directly addresses the course theme. Analyzes attack surfaces specific to Transformer FL: gradient inversion attacks that reconstruct training text/images from shared gradients, backdoor attacks exploiting the attention mechanism's global receptive field, and fairness issues arising from data heterogeneity across clients. Reviews defenses including robust aggregation, gradient clipping, and certified DP guarantees.

### Section 7 — Applications
Reviews domain-specific deployments: federated NLP (cross-hospital clinical NLP, cross-device keyboard prediction), federated ViT for medical imaging, multimodal FL, and emerging scientific use cases including federated spectral analysis — directly relevant to Raman spectroscopy workflows where laboratories cannot share raw spectra.

### Section 8 — Benchmarks and Evaluation
Catalogs available experimental infrastructure: FL simulation frameworks, standard non-IID data partitioning schemes, and Transformer-specific evaluation dimensions (convergence speed, communication rounds to target accuracy, privacy-utility tradeoff curves).

### Section 9 — Open Challenges
Identifies unsolved problems: the absence of theoretical convergence analysis for federated fine-tuning of large models, the gap between FL simulation and real deployment, and the need for trustworthiness certification frameworks.

### Section 10 — Conclusion
Summarizes the key findings, the proposed taxonomy, and the survey's contribution to structuring a fast-moving field.

---

## 4. Key Dimensions and Taxonomies

### Taxonomy 1: Training Paradigm
| Paradigm | Description | Representative Work |
|---|---|---|
| Full fine-tuning | All parameters communicated | FedBERT, FedRoBERTa |
| PEFT | Only adapter/LoRA params shared | FedPETuning, FLoRA |
| Prompt tuning | Only prompt embeddings shared | FedPrompt |
| Distillation-based | Only logits/soft labels shared | FedDF, DS-FL |
| Split learning | Model split between client/server | SplitFed |

### Taxonomy 2: Attention Mechanism Treatment
| Approach | Privacy Guarantee | Efficiency Impact |
|---|---|---|
| Standard attention, DP gradients | ε-DP on gradients | Moderate |
| Linearized/kernel attention | None built-in, decomposable | High |
| SMPC-based attention | Cryptographic | Very low |
| Federated attention distillation | None built-in | High |

### Taxonomy 3: Trustworthiness Dimension
- **Privacy**: DP noise level (ε, δ), gradient inversion resistance, membership inference resistance  
- **Robustness**: Byzantine fraction tolerance, backdoor detection rate  
- **Fairness**: Per-client accuracy variance, worst-group performance  
- **Accountability**: Auditability of model updates, contribution tracking  

### Taxonomy 4: Data Heterogeneity Setting
- IID (baseline)  
- Label skew (Dirichlet partitioning, pathological non-IID)  
- Feature shift (domain shift, covariate shift)  
- Quantity skew (highly unequal client dataset sizes)  

### Taxonomy 5: Application Domain
- NLP / LLMs  
- Computer vision (ViT)  
- Multimodal  
- Scientific/spectral data  
- Time series / sensors  

---

## 5. Key State-of-the-Art Papers to Review

### Federated Learning Foundations
- McMahan et al. (2017). *Communication-Efficient Learning of Deep Networks from Decentralized Data*. AISTATS. [FedAvg]
- Li et al. (2020). *Federated Optimization in Heterogeneous Networks*. MLSys. [FedProx]
- Karimireddy et al. (2020). *SCAFFOLD: Stochastic Controlled Averaging for Federated Learning*. ICML.
- Reddi et al. (2021). *Adaptive Federated Optimization*. ICLR. [FedOpt]

### Transformers in Federated Learning
- Zhuang et al. (2021). *Collaborative Unsupervised Visual Representation Learning from Decentralized Data*. ICCV. [early federated ViT]
- Lin et al. (2022). *FedPETuning: When Federated Learning Meets the Parameter-Efficient Tuning Methods of Pre-trained Language Models*. ACL Findings.
- Zhao et al. (2023). *FLoRA: Low-Rank Adapters Are Secretly Gradient Compressors*. ICML. [LoRA + FL connection]
- Che et al. (2023). *Federated Learning of Large Language Models with Parameter-Efficient Prompt Tuning and Adaptive Optimization*. EMNLP.
- Kuang et al. (2024). *FederatedScope-LLM: A Comprehensive Package for Fine-tuning Large Language Models in Federated Learning*. KDD.

### Privacy-Preserving Attention
- Zeng et al. (2023). *MPCFormer: Fast, Performant and Private Transformer Inference with MPC*. ICLR. [SMPC for Transformer inference]
- Li et al. (2022). *Mist: Defending Against Membership Inference Attacks Through Membership-Invariant Subspace Training*. [membership inference for Transformers]
- Shi et al. (2022). *Selective Differential Privacy for Language Modeling*. NAACL.

### Attacks and Robustness
- Geiping et al. (2020). *Inverting Gradients — How Easy Is It to Break Privacy in Federated Learning?* NeurIPS. [gradient inversion]
- Bagdasaryan et al. (2020). *How To Backdoor Federated Learning*. AISTATS.
- Xu et al. (2022). *Towards Building a Robust Toxicity Predictor*. ACL. [robustness of fine-tuned Transformers]

### Communication Efficiency
- Agarwal et al. (2018). *cpSGD: Communication-Efficient and Differentially-Private Distributed SGD*. NeurIPS.
- Lin et al. (2020). *Don't Use Large Mini-Batches, Use Local SGD*. ICLR.
- Diao et al. (2021). *HeteroFL: Computation and Communication Efficient Federated Learning for Heterogeneous Clients*. ICLR.

### Knowledge Distillation for FL
- Shen et al. (2020). *Federated Mutual Learning*. arXiv.
- Cronus / FedDF: Lin et al. (2020). *Ensemble Distillation for Robust Model Fusion in Federated Learning*. NeurIPS.

### Scientific / Spectroscopy Applications
- Lussier et al. (2020). *Deep learning and artificial intelligence methods for Raman and SERS analysis*. TrAC Trends in Analytical Chemistry.
- He et al. (2023). *Spectral Transformers for scientific data analysis*. (Representative of emerging direction.)
- Federal learning in healthcare / across labs: Rieke et al. (2020). *The Future of Digital Health with Federated Learning*. npj Digital Medicine.

---

## 6. Datasets, Benchmarks, and Evaluation Settings

### Standard FL Benchmark Datasets
| Dataset | Domain | Non-IID Partition | Use in Survey |
|---|---|---|---|
| FEMNIST | Handwriting (image) | By writer | Vision Transformer baseline |
| Shakespeare | NLP (next-char prediction) | By character | Language model FL |
| CelebA | Face attributes (image) | By identity | ViT fairness evaluation |
| CIFAR-10/100 | Image classification | Dirichlet (α) | ViT efficiency comparison |
| AG News / 20 Newsgroups | Text classification | By topic | Federated BERT fine-tuning |
| GLUE / SuperGLUE | NLP benchmarks | Simulated splits | FedPETuning evaluation |

### Spectroscopy / Scientific Datasets (relevant to Raman application)
| Dataset | Description | Availability |
|---|---|---|
| RRUFF Raman Database | ~5000 mineral Raman spectra | Public (rruff.info) |
| SERS pathogen spectra | Surface-enhanced Raman for bacteria classification | Various publications |
| UCI spectroscopy datasets | Multiple spectral domains | UCI ML Repository |

### FL Simulation Frameworks
- **LEAF** (Caldas et al., 2019): Federated benchmark suite, includes FEMNIST, Shakespeare
- **FedML** (He et al., 2020): Supports heterogeneous FL with Transformer models
- **Flower (flwr)**: Production-grade FL framework with PyTorch/HuggingFace integration
- **FederatedScope** (Xie et al., 2022): Includes FederatedScope-LLM for Transformer FL
- **PySyft**: Focused on privacy-preserving FL (DP, SMPC)
- **OpenFL** (Intel): Cross-silo FL

### Evaluation Dimensions
1. **Accuracy**: Top-1 / F1 at convergence vs. centralized baseline
2. **Communication cost**: Total bits transmitted to reach target accuracy
3. **Convergence speed**: Rounds to reach target accuracy
4. **Privacy**: (ε, δ)-DP guarantee, empirical membership inference AUC
5. **Robustness**: Accuracy under X% Byzantine clients, backdoor attack success rate
6. **Fairness**: Variance in per-client accuracy, worst-10% client performance
7. **Compute**: Per-round client FLOPs (important for edge deployment)

### Non-IID Partitioning Standards
- **Dirichlet (Dir(α))**: α → 0 = extreme heterogeneity; α → ∞ = IID. Standard in recent FL literature.
- **Pathological**: Each client holds only K classes (e.g., K=2 of 10).
- **Natural splits**: Use dataset-inherent structure (e.g., by user/device/institution).

---

## 7. Proposed Evaluation Protocol for the Survey

To ensure the survey is actionable, a reproducibility checklist will be applied to all reviewed papers:

- [ ] Is the non-IID partition method specified?
- [ ] Is the number of clients and participation rate stated?
- [ ] Are communication costs reported?
- [ ] Are privacy guarantees formally stated (if claimed)?
- [ ] Is the code publicly available?
- [ ] Are results compared against FedAvg baseline?

---

## 8. Connection to Raman Spectroscopy (Personal Research Angle)

Raman spectroscopy produces 1D sequential intensity spectra indexed by wavenumber — a natural input for Transformer sequence models (treating spectral bins as tokens). In a federated scenario, multiple research laboratories or hospitals each hold proprietary Raman datasets (e.g., for cancer tissue classification, mineral identification, or pharmaceutical quality control) and cannot share raw spectra due to IP or regulatory constraints.

This creates a direct use case for federated Transformer training on spectral data, and could serve as a running example / case study throughout the survey, grounding abstract concepts in a concrete scientific application.

**Specific questions this angle raises:**
- Can a shared Transformer backbone trained federally across Raman labs generalize better than local models?
- Do attention heads learn interpretable spectral features (peak positions, bandwidths)?
- What are the privacy risks of sharing gradient updates derived from proprietary spectra?

---

*Proposal prepared for the PhD Mini-course on Ensuring Trustworthiness in Federated Learning.*
