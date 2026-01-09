# Thesis Viva Preparation: Questions & Answers

## Comparative Analysis of RoBERTa and CodeBERT for Automated Misconfiguration Detection in Dockerfiles

---

## 1. Why 256 tokens as max length? Did you experiment with other lengths?

The 256-token limit was chosen based on **empirical analysis of Dockerfile instruction lengths** and **computational constraints**.

When analyzing our dataset of ~1.6 million instructions, we found that the vast majority of individual Dockerfile commands (like `RUN apt-get install...`, `COPY src/ /app/`) fit comfortably within 256 tokens. Since our approach processes instructions at the **line level** rather than entire Dockerfiles, this length captures sufficient context without excessive padding.

Additionally, 256 tokens provides a good **balance between model capacity and computational efficiency** — doubling to 512 tokens would quadruple memory usage due to the quadratic complexity of self-attention (O(n²)), while providing diminishing returns for single-line instructions.

While we didn't extensively experiment with other lengths due to resource constraints, in preliminary tests with 128 tokens, we observed truncation issues with longer RUN commands. The 256-token setting is also consistent with common practice in code-related NLP tasks.

---

## 2. How does your approach compare to existing tools like Hadolint?

Our approach and Hadolint are **complementary rather than competing** solutions:

| Aspect | Hadolint | Our NLP Approach |
|--------|----------|------------------|
| **Mechanism** | Rule-based static analysis using predefined regex patterns | Learning-based using transformer models trained on labeled examples |
| **Flexibility** | Detects only explicitly programmed rules | Can potentially generalize to **unseen patterns** based on learned representations |
| **Explainability** | Highly explainable — points to specific rule violated | Less transparent (black-box), though we map predictions to rule IDs |
| **Maintenance** | Requires manual rule updates as best practices evolve | Can be retrained on new data to adapt automatically |
| **Coverage** | Limited to ~100 predefined rules | Trained on 63 rules from our dataset, but model can learn semantic patterns |

The key advantage of our approach is that it **learns contextual patterns** that rule-based tools might miss. For example, a transformer can understand that `apt-get install` without `--no-install-recommends` in a production image is problematic, even if the exact command differs from the hardcoded rule pattern.

Ideally, a **hybrid approach** combining Hadolint's precision with our model's semantic understanding would provide the best of both worlds — which is why we included this in our future recommendations.

---

## 3. Why only 3 epochs? Did you observe overfitting?

We chose 3 epochs based on **established best practices for fine-tuning pretrained transformers** and our validation monitoring.

Pretrained models like RoBERTa and CodeBERT have already learned rich language representations from millions of examples. Fine-tuning requires only a few epochs to adapt these representations to our specific task — more epochs risk **catastrophic forgetting** where the model loses its pretrained knowledge.

During training, we monitored **validation loss at each epoch**. We observed that:
- Loss decreased significantly in epoch 1
- Continued improving in epoch 2
- Showed minimal improvement or slight increase in epoch 3

This pattern indicated we were approaching the optimal point. Running beyond 3 epochs showed signs of overfitting on preliminary experiments, with training loss continuing to decrease while validation loss started increasing.

---

## 4. How would you handle completely new/unseen rule violations?

This is an important limitation of our current approach. Since our model is trained on 63 known rule categories, it **cannot directly classify a completely new rule type** it has never seen.

However, our **two-stage architecture provides partial resilience**:

1. **Stage 1 (Binary Classification)** can still detect that *something is wrong* — it learns general patterns of misconfiguration, not just specific rules. So a new violation might still be flagged as 'Wrong' even if we can't categorize it.

2. For **truly novel violations**, the model would likely:
   - Misclassify it under the most semantically similar known rule, OR
   - Show low confidence scores, which could trigger manual review

**Practical solutions include:**
- Periodic retraining with updated datasets as new rules emerge
- Implementing a **confidence threshold** — predictions below a threshold get flagged for human review
- Hybrid approach: use rule-based tools for known patterns, NLP model for semantic anomalies

---

## 5. Why did you choose instruction-level (line-level) processing instead of entire Dockerfile?

We chose instruction-level processing for several reasons:

1. **Granularity of violations**: Dockerfile misconfigurations typically occur at the individual instruction level. A single `RUN` command might violate a rule while the rest of the Dockerfile is correct. Line-level analysis provides **precise localization** of issues.

2. **Token limit constraints**: Entire Dockerfiles can be very long (hundreds of lines). Processing them would require either truncation (losing information) or very long sequence lengths (computationally expensive).

3. **Data augmentation**: Line-level processing naturally provides more training samples from the same Dockerfile, giving us ~1.6 million instruction samples from ~178,800 Dockerfiles.

4. **Practical utility**: In real CI/CD integration, developers want to know *which specific line* has the issue, not just that *somewhere* in the Dockerfile there's a problem.

The tradeoff is losing some **cross-instruction context** — for example, understanding that a `COPY` depends on a previous `WORKDIR`. Future work could explore hierarchical models that combine line-level and file-level understanding.

---

## 6. Why CodeBERT over other code-specific models like GraphCodeBERT or UniXcoder?

CodeBERT was selected as a **strong baseline code-aware model** for several reasons:

1. **Proven effectiveness**: CodeBERT has demonstrated strong performance across multiple code understanding tasks and is well-established in the research community.

2. **Architecture compatibility**: CodeBERT shares the same architecture as RoBERTa, allowing **fair comparison** between a general-purpose model and a code-specialized model under identical conditions.

3. **Computational feasibility**: More advanced models like GraphCodeBERT require additional structural information (AST, data flow graphs), which is non-trivial to extract from Dockerfile instructions. UniXcoder, while powerful, requires more computational resources.

4. **Research scope**: Our primary research question was whether **code-aware pretraining provides advantages over general text pretraining** for Dockerfile analysis. CodeBERT vs RoBERTa cleanly answers this question.

Future work could definitely explore more advanced architectures — this is mentioned in our recommendations.

---

## 7. What is the practical deployment scenario for your model?

The intended deployment scenario is **integration into CI/CD pipelines** for automated Dockerfile quality checks:

1. **Pre-commit hook**: Developers receive instant feedback before pushing code
2. **Pull Request check**: Automated review flagging potential issues
3. **Container registry gate**: Block deployment of images built from problematic Dockerfiles

**Inference workflow:**
1. Parse Dockerfile into individual instructions
2. Run each instruction through Stage 1 (binary classification)
3. For instructions flagged as 'Wrong', run Stage 2 to identify the specific rule violated
4. Generate a report with line numbers and suggested fixes

With proper optimization (quantization, batching), inference takes **milliseconds per instruction**, making it suitable for real-time feedback.

---

## 8. How did you handle class imbalance in your dataset?

Class imbalance was a significant challenge, addressed differently in each stage:

**Stage 1 (Binary Classification):**
- Original dataset was heavily imbalanced with more 'Correct' than 'Wrong' instructions
- We **downsampled to 200,000 samples per class**, ensuring perfect 50-50 balance
- This prevents the model from learning a trivial 'always predict majority class' strategy

**Stage 2 (Multi-class / Rule Classification):**
- Some rules had thousands of examples while others had fewer than 100
- We applied **two-sided filtering**:
  - Removed rules with <200 samples (insufficient for learning)
  - Capped rules at ≤5,000 samples (prevent dominant rules from overwhelming)
- Used **stratified splitting** to ensure all 63 final rules appear proportionally in train/val/test sets

The result: a more balanced dataset where the model learns meaningful patterns for each rule rather than memorizing frequency distributions.

---

## 9. What are the main limitations of your research?

I acknowledge several limitations:

1. **Computational constraints**: We used a subset (~400K samples) of the full 1.6M dataset. Training on the complete dataset might yield better results.

2. **Rule coverage**: We trained on 63 rules after filtering. Some rare but important rules were excluded, and completely novel misconfigurations cannot be detected.

3. **Context limitation**: Instruction-level processing loses cross-line context. Some misconfigurations require understanding relationships between multiple instructions.

4. **Explainability**: Transformer models are black-boxes. While we predict rule IDs, we cannot explain *why* a specific instruction triggered that rule.

5. **Evaluation scope**: We evaluated on the same distribution as training. Real-world Dockerfiles may have different characteristics or domain-specific patterns.

6. **No direct comparison with baselines**: While we discussed Hadolint conceptually, we didn't run direct comparative experiments on the same test set.

---

## Quick Reference: Key Metrics

| Metric | RoBERTa | CodeBERT |
|--------|---------|----------|
| Binary Classification Accuracy | 98% | 98% |
| Binary F1-Score | 0.98 | 0.98 |
| Binary AUC-ROC | 0.9968 | 0.9970 |
| Multi-class Top-1 Accuracy | 70.48% | 70.70% |
| Multi-class Top-3 Accuracy | 89.01% | 89.38% |
| Macro-averaged F1 | 0.7055 | 0.7136 |

---

