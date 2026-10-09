## Red Barrel Segmentation with Gaussian Discriminant Analysis

Pixel-level detection of red barrels in real-world images, using a GDA classifier on raw RGB values. Two variants are compared: a **shared covariance** matrix across classes and **class-specific** covariance matrices.

### Approach

1. **Labeling:** Hand-labeled regions of interest with `roipoly`. Each training image has a positive ROI (the barrel) and a similarly sized negative ROI (background), which keeps the priors balanced ($\pi_0 \approx \pi_1$). Test images have only the barrel traced, as ground truth.
2. **Training:** For each class $k$, estimate the prior $\pi_k = n_k / n$, mean $\mu_k$, and covariance $\Sigma_k$. The shared covariance is the sample-size-weighted average $\Sigma = \sum_k n_k \Sigma_k / \sum_k n_k$.
3. **Classification:** Each pixel $x$ gets the label that maximizes the log posterior (dropping terms shared across classes):

$$
\hat{y} = \arg\max_{k \in \{0,1\}} \left[ -\tfrac{1}{2}(x-\mu_k)^\top \Sigma_k^{-1}(x-\mu_k) - \tfrac{1}{2}\log|\Sigma_k| + \log \pi_k \right]
$$

   With a shared covariance, the $\log|\Sigma|$ term cancels and the decision boundary is linear.
4. **Segmentation:** Predicted barrel pixels are rendered red, background black.

### Results

| Model | Class | Precision | Recall |
|---|---|---|---|
| Shared covariance | Background (0) | 99.83% | 96.45% |
| | Barrel (1) | 42.85% | 94.06% |
| Class-specific covariance | Background (0) | 99.89% | 95.22% |
| | Barrel (1) | 36.32% | 96.28% |

Both models recover over 94% of barrel pixels. Class-specific covariance gains about 2 points of barrel recall, at the cost of barrel precision.

### Takeaways

- **Color-only features cause false positives.** Other red objects (e.g., a red vending machine) are classified as barrel, which explains the low barrel precision.
- **Labeling affects results.** Hand-drawn negative ROIs can introduce bias; a more randomized negative sampling strategy could help.
- **Possible improvements:** add spatial or texture features, or sample negatives more randomly.
