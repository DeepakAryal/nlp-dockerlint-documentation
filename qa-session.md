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

## 10. Can you explain the self-attention formula in the transformer encoder?

The self-attention mechanism is defined by the formula:

**Attention(Q, K, V) = softmax(QK^T / √d_k) V**

Let me break down each component:

**Components:**
- **Q (Query)**: Represents what we're looking for in the sequence (dimension: n × d_k)
- **K (Key)**: Represents what each token offers as information (dimension: n × d_k)  
- **V (Value)**: The actual information/content to extract (dimension: n × d_v)
- **d_k**: Dimension of the key vectors (also query dimension)
- **n**: Sequence length (number of tokens)

**Step-by-step computation:**

1. **QK^T**: Matrix multiplication producing attention scores (n × n matrix)
   - Each element (i,j) represents how much token i should attend to token j
   - Shows compatibility between queries and keys

2. **Scaling by √d_k**: Prevents dot products from becoming too large
   - Without scaling, large d_k values lead to very small gradients after softmax
   - √d_k scaling keeps variance stable regardless of dimension

3. **Softmax**: Converts scores into probability distribution
   - Each row sums to 1
   - High scores get exponentially more weight
   - Formula: softmax(x_i) = exp(x_i) / Σ exp(x_j)

4. **Multiply by V**: Weighted sum of values
   - Tokens with high attention scores contribute more to the output
   - Produces contextualized representation for each token

**In our Dockerfile context:**
For a RUN instruction like `RUN apt-get update && apt-get install -y python3`, the self-attention mechanism allows:
- The token "install" to attend to both "apt-get" and "python3"
- Understanding the entire command context, not just adjacent words
- Capturing long-range dependencies that simple sequential models might miss

---

## 11. Why do we divide by √d_k in the attention formula?

This is a crucial normalization step to maintain **gradient stability** during training.

**The problem without scaling:**
- Dot product QK^T has variance proportional to d_k (dimension of key vectors)
- For large d_k (e.g., 768 in our models), dot products can become very large in magnitude
- When fed into softmax, large values cause the function to push probabilities toward 0 and 1 (saturating)
- Saturated softmax has extremely small gradients (near zero)
- This causes **vanishing gradients**, making the model hard to train

**Mathematical intuition:**
If Q and K are random vectors with mean 0 and variance 1:
- Their dot product q·k = Σ(q_i × k_i) has variance ≈ d_k
- Standard deviation grows as √d_k
- Dividing by √d_k normalizes variance back to ~1

**Example with numbers:**
- If d_k = 64 and dot product = 100
- After scaling: 100/√64 = 100/8 = 12.5
- This keeps values in a reasonable range for softmax

In our RoBERTa/CodeBERT models (d_k = 64 per head), this scaling is essential for stable training across 12 attention heads and 12 layers.

---

## 12. Can you explain the classification head formula: y = softmax(W·h_CLS + b)?

This formula describes how we convert the encoder's output into class predictions.

**Components:**

1. **h_CLS**: The [CLS] token embedding from the final encoder layer
   - Dimension: d_model (768 for base models)
   - This special token aggregates information from the entire sequence
   - Acts as a sentence/instruction-level representation

2. **W (Weight matrix)**: Trainable linear transformation
   - Dimension: num_classes × d_model
   - For binary classification: 2 × 768
   - For rule classification: 63 × 768
   - Projects the 768-dimensional embedding to class logits

3. **b (Bias vector)**: Trainable offset term
   - Dimension: num_classes
   - Allows the model to learn class imbalances

4. **W·h_CLS + b**: Linear transformation producing logits
   - Output dimension: num_classes
   - These are raw, unnormalized scores for each class

5. **softmax**: Converts logits into probability distribution
   - Formula: softmax(z_i) = exp(z_i) / Σ_j exp(z_j)
   - Output: Probability distribution where all values sum to 1
   - Each y_i represents P(class_i | input)

**Concrete example for binary classification:**

Input: `RUN apt-get install python` (potential misconfiguration)

1. h_CLS from encoder = [768-dimensional vector]
2. W·h_CLS + b = [logit_correct, logit_wrong] = [-2.3, 4.1]
3. softmax application:
   - exp(-2.3) ≈ 0.10, exp(4.1) ≈ 60.3
   - P(correct) = 0.10 / (0.10 + 60.3) ≈ 0.002
   - P(wrong) = 60.3 / (0.10 + 60.3) ≈ 0.998

Prediction: **Wrong** (98% confidence)

**For rule classification (63 classes):**
- Same process but W has 63 rows instead of 2
- Outputs 63 probabilities, one per rule
- We take argmax to get the predicted rule ID

This is a standard fully-connected layer + softmax, commonly called a "classification head" in transfer learning.

---

## 13. What is the role of the [CLS] token and how does it aggregate sequence information?

The **[CLS]** (classification) token is a special token that serves as the sequence-level representation.

**Mechanism:**

1. **Position**: Always inserted at the **beginning** of the input sequence
   - Example: `[CLS] RUN apt-get install python`

2. **Initial embedding**: Starts with a learned embedding (no inherent meaning)

3. **Information aggregation via self-attention**:
   - In each encoder layer, [CLS] attends to all other tokens
   - Through the attention mechanism: Attention(Q, K, V)
   - [CLS] query vector attends to keys from all instruction tokens
   - It receives a weighted combination of all token representations

4. **Multi-layer refinement**:
   - Layer 1: [CLS] learns basic token interactions
   - Layer 6: [CLS] captures mid-level patterns
   - Layer 12: [CLS] contains high-level semantic understanding

5. **Final representation**: After 12 transformer layers:
   - h_CLS contains aggregated information about the entire instruction
   - Encodes whether the instruction is correct/wrong and which rule applies

**Why use [CLS] instead of averaging all tokens?**
- **Learned aggregation**: The model learns *how* to aggregate (via attention weights), not a fixed average
- **Task-specific**: Attention patterns optimize for classification during training  
- **Flexibility**: Can focus on relevant tokens, ignore padding/irrelevant words

**In our models:**
- RoBERTa and CodeBERT both use the [CLS] token following BERT architecture
- We extract h_CLS (768-dim vector) from the final layer
- Pass it through the classification head: W·h_CLS + b

---

## 14. How does positional encoding work in transformers and why is it needed?

Positional encoding is essential because **transformers have no inherent notion of token order**.

**The problem:**
Unlike RNNs/LSTMs that process sequences step-by-step, self-attention is a **permutation-invariant** operation:
- Attention(Q, K, V) produces the same result regardless of token order
- Without position info, "RUN apt-get install" = "install apt-get RUN" (clearly wrong!)

**Solution: Add positional information to embeddings**

**Original Transformer (Vaswani et al.) uses sinusoidal encoding:**

PE(pos, 2i) = sin(pos / 10000^(2i/d_model))  
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

Where:
- pos: Position in sequence (0, 1, 2, ...)
- i: Dimension index (0 to d_model/2)
- d_model: Embedding dimension (768)

**Properties:**
- Different frequencies for different dimensions
- Allows model to learn relative positions
- Deterministic (not learned)

**RoBERTa/CodeBERT use learned positional embeddings:**
- Each position (0 to 511) has a learnable embedding vector
- These are added to token embeddings at the input layer
- More flexible, optimized during pretraining

**In our case:**
- Max sequence length: 256 tokens
- Each position 0-255 has a learned 768-dim vector
- Final input = Token_Embedding + Positional_Embedding

**Example:**
Token sequence: `[CLS] RUN apt-get install`

Position 0: [CLS] embedding + position_0 embedding  
Position 1: RUN embedding + position_1 embedding  
Position 2: apt-get embedding + position_2 embedding  
...

This allows the model to understand that in Dockerfiles, order matters (e.g., `FROM` should come first).

---

## 15. What's the difference between encoder-only and encoder-decoder transformers?

**Architectural differences:**

| Aspect | Encoder-only (BERT, RoBERTa, CodeBERT) | Encoder-Decoder (T5, BART) |
|--------|----------------------------------------|---------------------------|
| **Structure** | Only encoding layers with bidirectional attention | Encoder + decoder with cross-attention |
| **Attention** | Each token sees all tokens (bidirectional) | Encoder: bidirectional; Decoder: causal (unidirectional) |
| **Use case** | Classification, understanding tasks | Generation, translation, summarization |
| **Our task** | ✓ Perfect for classification | ✗ Overkill, adds unnecessary complexity |

**Why we use encoder-only for our task:**

1. **Classification nature**: We need to understand the input and assign a label, not generate new text
2. **Bidirectional context**: We want each token to see the full instruction context (before and after)
3. **Efficiency**: Encoder-only is faster and simpler for classification
4. **Pretrained models**: RoBERTa and CodeBERT are already optimized encoder-only architectures

**When encoder-decoder would be useful:**
- If we wanted to generate Dockerfile **corrections** (e.g., input: wrong instruction → output: fixed instruction)
- Translating natural language requirements to Dockerfiles
- Generating explanations for why a rule was violated

For our binary and multi-class classification tasks, encoder-only is the optimal choice.

---

## 16. Can you explain what happens in one transformer encoder layer?

A single transformer encoder layer processes input through two main sub-layers:

**Architecture of one encoder layer:**

```
Input: X (n × d_model)
    ↓
[1] Multi-Head Self-Attention
    ↓
Add & Normalize (Residual connection)
    ↓
[2] Feed-Forward Network
    ↓
Add & Normalize (Residual connection)
    ↓
Output: X' (n × d_model)
```

**Detailed breakdown:**

**[1] Multi-Head Self-Attention:**
- Splits d_model dimensions into h heads (RoBERTa: 12 heads)
- Each head: d_k = d_model / h = 768 / 12 = 64 dimensions
- Computes Attention(Q, K, V) = softmax(QK^T/√d_k)V for each head in parallel
- Concatenates all head outputs and applies linear projection
- Allows the model to attend to different aspects simultaneously

**Add & Normalize (LayerNorm):**
- Residual connection: X + Attention(X)
- Layer normalization: normalizes across features for each token
- Prevents gradient vanishing, stabilizes training

**[2] Feed-Forward Network (FFN):**
- Two linear transformations with activation:
  - FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
  - First layer expands: 768 → 3072 dimensions
  - Second layer projects back: 3072 → 768
- Applied identically to each position (shared across all tokens)
- Adds non-linear transformation capacity

**Second Add & Normalize:**
- Residual: X + FFN(X)
- Layer normalization again

**In our RoBERTa/CodeBERT (12 layers):**
- Layer 1-4: Learn basic patterns (syntax, keywords)
- Layer 5-8: Intermediate representations (command structure)
- Layer 9-12: High-level semantics (misconfiguration patterns)

Each layer refines the representation, building progressively abstract understanding of the Dockerfile instruction.

---

## 17. What is dynamic masking in RoBERTa and how does it differ from BERT?

Dynamic masking is one of RoBERTa's key improvements over BERT during **pretraining** (not affecting our fine-tuning directly, but improving the base model).

**BERT's static masking:**
- During preprocessing, 15% of tokens are randomly masked ONCE
- The same masked version is used for all epochs
- Example: "RUN apt-get install python"
  - Epoch 1, 2, 3, ..., all training: "RUN [MASK] install python"
- Model sees the same masked pattern repeatedly

**RoBERTa's dynamic masking:**
- Masking is applied **on-the-fly** during training
- Each time a sequence is sampled, a different 15% is masked
- Example: "RUN apt-get install python"
  - Epoch 1: "RUN [MASK] install python"
  - Epoch 2: "RUN apt-get [MASK] python"
  - Epoch 3: "[MASK] apt-get install python"
- Model learns from more diverse masking patterns

**Benefits for downstream tasks (like ours):**
1. **Better generalization**: Model has seen tokens in more varied contexts
2. **Reduced overfitting**: Less memorization of specific mask positions
3. **Improved representations**: More robust token embeddings

**Why this matters for our Dockerfile task:**
Even though we don't use masking during fine-tuning:
- RoBERTa's pretrained weights are **more robust** due to dynamic masking
- Better initialization for our classification task
- Explains why RoBERTa often outperforms BERT on downstream tasks

**Other RoBERTa improvements beyond dynamic masking:**
- Trains on 10× more data than BERT
- Larger batch sizes
- Removes Next Sentence Prediction (NSP) objective
- Trains longer

These collectively make RoBERTa a stronger baseline model for our task.

---

## 18. You get only 1.15% better CodeBERT result in macro F1 average. Can we say CodeBERT is better with just 1.15% improvement?

While the 1.15% macro F1 improvement might seem modest at first glance, the recommendation for CodeBERT rests on three pillars:

**First, systematic superiority** — CodeBERT outperforms RoBERTa across all five evaluation metrics, not just one. This indicates a fundamental advantage from code-aware pre-training rather than random variance.

**Second, task complexity** — this improvement is on a 63-class imbalanced classification task. In multi-class problems of this complexity, improvements of 1-2% are considered significant in the literature, especially when the baseline is already strong at 70% F1.

**Third, practical impact** — in production environments processing thousands of Dockerfiles, this translates to hundreds of additional misconfigurations detected annually, directly improving software quality and security.

Most importantly, the scientific contribution is demonstrating that code-specific pre-training provides measurable advantages over general-purpose models. If we ignore these results and recommend the inferior model, we would be contradicting our own empirical evidence.
