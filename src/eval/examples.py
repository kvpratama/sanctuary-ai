"""Evaluation examples for the sanctuary dataset.

Each example contains inputs (question, document_id, user_id),
expected outputs (answer), and metadata (external_id).
"""

from typing import Any

examples: list[dict[str, Any]] = [
    {
        "inputs": {
            "question": "How do I handle data distribution shifts in a live environment?",
            "document_id": "1ed41248-cf8a-4231-8a33-415191dc9c31",
            "user_id": "f9937aab-6c97-4c3e-a6f8-38f4a1676200",
        },
        "outputs": {
            "answer": """In a live environment, the first step is **detecting shifts** by monitoring artifacts like **model predictions and input features**, as ground truth labels are often delayed or unavailable. Because predictions are generally low-dimensional, they serve as an effective proxy for input shifts; a significant change in the prediction distribution usually indicates that the underlying data distribution has diverged from the training set. It is crucial to **rule out internal human errors** first, such as bugs in the data pipeline, missing values, or mismatched schemas, which frequently mimic distribution shifts on monitoring dashboards. **Observability** further aids this diagnosis by allowing teams to slice metrics along different dimensions to identify exactly which subgroups of users or types of inputs are driving the performance degradation.

Once a shift is confirmed, the primary remediation is **retraining the model** using labeled data from the new target distribution. This can be achieved through **stateless retraining**, where the model is rebuilt from scratch using both old and new data, or **stateful training (fine-tuning)**, which continues training the existing model on fresh data to converge faster and save compute resources. Beyond reactive retraining, systems can be **proactively designed for robustness** by choosing stable features, such as bucketing rapidly changing raw rankings into broader categories. Additionally, maintaining **separate models for different data slices** (e.g., different geographic regions) allows for localized adaptation, ensuring that a fast-changing market segment can be updated frequently without affecting more stable segments.

For sustainable management, organizations should move toward **continual learning infrastructure** to automate model updates triggered by time, performance drops, or detected drifts. Before finalizing an update schedule, it is vital to **quantify the value of data freshness** through experiments that measure exactly how much model performance improves when using more recent data. Finally, any updated "challenger" model must be **tested in production** safely before a full rollout using techniques like **shadow deployment**, where predictions are logged but not served to users, or **A/B testing and canary releases** to evaluate the new model's performance against the current "champion" model in a live setting.""",
        },
        "metadata": {
            "external_id": "id_1",
        },
    },
    {
        "inputs": {
            "question": "How do I ensure my model performs as well in production as it did during training?",
            "document_id": "1ed41248-cf8a-4231-8a33-415191dc9c31",
            "user_id": "f9937aab-6c97-4c3e-a6f8-38f4a1676200",
        },
        "outputs": {
            "answer": """To ensure your model performs as well in production as it did during training, you must first address **train-serving skew**, a common failure mode where a model generalizes poorly because the underlying distribution of real-world data diverges from your finite training data. **Offline evaluation** should go beyond aggregate accuracy metrics to include **perturbation tests**, which simulate noisy production inputs, and **invariance tests** to ensure that changes to sensitive information do not lead to incorrect outputs. It is also essential to evaluate your model against various **baselines**, such as human experts or simple heuristics, to justify the complexity of the ML solution.

Methodologically, you must prevent **data leakage** by splitting time-correlated data by time rather than randomly and ensuring that **feature scaling** statistics are derived solely from the training split. Maintaining **feature consistency** is another critical factor, as having separate data pipelines for training and inference is a frequent source of production bugs. Implementing a **feature store** can mitigate this risk by unifying the logic for both batch and streaming features, ensuring that the model receives the same inputs during inference as it did during development.

Once the model is ready for deployment, use **shadow deployment** to route live traffic to the new model without serving its predictions to users, allowing you to safely analyze its performance against the existing "champion" model. Techniques such as **A/B testing and canary releases** further enable you to validate the model's effectiveness with live data before a full rollout. Finally, establishing **robust monitoring and observability** is required to detect "silent failures," where the system remains operational but prediction quality degrades due to constantly shifting data distributions.""",
        },
        "metadata": {
            "external_id": "id_2",
        },
    },
    {
        "inputs": {
            "question": "How do I detect and fix a 'silent failure' once the model is deployed?",
            "document_id": "1ed41248-cf8a-4231-8a33-415191dc9c31",
            "user_id": "f9937aab-6c97-4c3e-a6f8-38f4a1676200",
        },
        "outputs": {
            "answer": """**Silent failures** are uniquely challenging because the system continues to run without throwing operational errors (like 404s or crashes), producing incorrect predictions that users may trust blindly. To detect these, you should primarily **monitor model predictions**; because they are low-dimensional, sudden changes in their distribution often serve as an effective proxy for shifts in the underlying input data. Additionally, implementing **feature validation** to ensure inputs follow expected schemas and tracking **user feedback**—such as click-through or completion rates—can reveal performance drops that aggregate metrics might hide. **Observability** is crucial here, as it allows you to "slice and dice" metrics by user subgroups or time windows to identify if a failure is global or isolated to specific inputs.

Once a failure is detected, you must first **diagnose the root cause**, as a large percentage of perceived data shifts are actually **internal human errors** like pipeline bugs, missing values, or inconsistent feature extraction between training and inference. It is vital to rule out these operational issues before assuming a fundamental change in the environment has occurred. If the failure is indeed caused by a **data distribution shift**, such as covariate shift or concept drift, you must adapt the model to the new target distribution. **Statistical methods**, including two-sample hypothesis tests like the Kolmogorov-Smirnov test, can help confirm if the difference between production data and training data is statistically significant enough to merit intervention.

To fix these failures and increase future reliability, the most common industry approach is **retraining the model** with fresh labeled data from the target distribution, utilizing either **stateless retraining** (from scratch) or **stateful training (fine-tuning)** to save compute resources. Beyond reactive retraining, you can **design for robustness** by choosing stable features—for instance, bucketing rapidly changing raw rankings into broader categories. Implementing **smooth failing** with a backup system, such as a simple heuristic or a set of precomputed predictions, ensures the system remains useful if the main model produces suspicious outputs. Finally, to correct **degenerate feedback loops** that bias future training data, you can introduce **randomization** to explore new items or use positional features to decouple a model's prediction from its rank on a screen."""
        },
        "metadata": {
            "external_id": "id_3",
        },
    },
    {
        "inputs": {
            "question": "How do I handle data challenges like labels, imbalance, and distribution shifts?",
            "document_id": "1ed41248-cf8a-4231-8a33-415191dc9c31",
            "user_id": "f9937aab-6c97-4c3e-a6f8-38f4a1676200",
        },
        "outputs": {
            "answer": """To address the lack of high-quality labels, which are often slow and expensive to acquire manually, you can leverage **natural labels** derived from user behavioral signals or implement **programmatic labeling** through **weak supervision**. Weak supervision allows you to encode domain expertise into **labeling functions** to generate labels at scale, while **semi-supervision** and **transfer learning** minimize the need for massive initial datasets by utilizing structural assumptions or pretrained models. Furthermore, **active learning** can increase data efficiency by having the model proactively identify and query the most informative samples for human annotation.

**Class imbalance** is the norm in real-world settings and should be handled by first moving away from aggregate accuracy toward **appropriate evaluation metrics** like F1 score, precision, recall, and ROC curves. Effective remediation involves **data-level methods**, such as **resampling** (oversampling minority classes or undersampling majority ones), and **algorithm-level methods** that modify the **loss function**. For instance, **cost-sensitive learning** or **focal loss** can be used to punish the model more heavily for mistakes on rare classes, forcing it to prioritize the minority signal.

**Data distribution shifts**—such as covariate shift, label shift, and concept drift—occur when the target distribution diverges from the training distribution, leading to performance degradation. You can detect these shifts using **statistical methods** like two-sample hypothesis tests (e.g., the Kolmogorov-Smirnov test) to compare incoming production data against the training set. To handle confirmed shifts, the most common industrial approach is **retraining the model with fresh data** from the new distribution; alternatively, you can **design for robustness** by selecting stable features or using separate models for different data slices."""
        },
        "metadata": {
            "external_id": "id_4",
        },
    },
    {
        "inputs": {
            "question": "What best describes the relationship between MLOps and Machine Learning Systems Design?",
            "document_id": "1ed41248-cf8a-4231-8a33-415191dc9c31",
            "user_id": "f9937aab-6c97-4c3e-a6f8-38f4a1676200",
        },
        "outputs": {
            "answer": """**ML systems design** is best described as a **system approach to MLOps**, focusing on the holistic coordination of all components and stakeholders to meet specific objectives. 

The relationship between the two can be understood through their respective focuses:

*   **MLOps** is defined as a set of **tools and best practices** used to bring machine learning into production. This involves the "operationalization" of ML, which includes **deploying, monitoring, and maintaining** models in a live environment.
*   **Machine Learning Systems Design** provides the **principled, end-to-end view** necessary to navigate MLOps. It ensures that every part of the system—including **business requirements, the data stack, infrastructure, and deployment logic**—works together as a unified whole rather than in isolation.

While MLOps provides the technical "how" for productionizing models, ML systems design acts as the **map and compass**, considering the system holistically to satisfy requirements like **reliability, scalability, maintainability, and adaptability**. This holistic approach is necessary because changes in one component, such as an ML algorithm, will likely affect other components like the data pipeline or business logic."""
        },
        "metadata": {
            "external_id": "id_5",
        },
    },
    {
        "inputs": {
            "question": "How does the 'Software 2.0' approach (Machine Learning) differ from traditional software development in terms of handling patterns?",
            "document_id": "1ed41248-cf8a-4231-8a33-415191dc9c31",
            "user_id": "f9937aab-6c97-4c3e-a6f8-38f4a1676200",
        },
        "outputs": {
            "answer": """The primary difference is that traditional software requires developers to **hand-specify patterns** (rules) to calculate outputs, whereas the Software 2.0 approach **learns patterns from data**. While traditional software uses hardcoded rules that become outdated if conditions change, Software 2.0 can **automatically discover and adapt to complex patterns**—such as those in speech recognition or dynamic pricing—that are too difficult for humans to define manually."""
        },
        "metadata": {
            "external_id": "id_6",
        },
    },
    {
        "inputs": {
            "question": "Why might an ML model that achieves state-of-the-art performance in a research setting fail to be adopted in a production environment?",
            "document_id": "1ed41248-cf8a-4231-8a33-415191dc9c31",
            "user_id": "f9937aab-6c97-4c3e-a6f8-38f4a1676200",
        },
        "outputs": {
            "answer": """Research models often fail in production because they focus solely on **maximizing accuracy on static datasets**, while production environments require balancing conflicting stakeholder needs like **low latency, cost-effectiveness, and interpretability**. Many state-of-the-art techniques, such as ensembling, make systems **too complex and slow** to be useful for real-time applications where every millisecond of delay hurts user conversion rates. Models may also perform well in research due to **data leakage**—where information from the future "leaks" into training—only to fail spectacularly when that information is unavailable in the real world. Furthermore, models trained on clean, stationary research data often struggle to generalize to the **noisy, biased, and constantly shifting data distributions** encountered in production. Finally, the **massive computational resources and memory** required to run billion-parameter SOTA models can make them too expensive or technically difficult to deploy at scale."""
        },
        "metadata": {
            "external_id": "id_7",
        },
    },
    {
        "inputs": {
            "question": "In the context of latency distributions, why is the average (arithmetic mean) often a misleading metric for evaluating system performance?",
            "document_id": "1ed41248-cf8a-4231-8a33-415191dc9c31",
            "user_id": "f9937aab-6c97-4c3e-a6f8-38f4a1676200",
        },
        "outputs": {
            "answer": """Latency is a **distribution rather than a single number**, and simplifying it into an arithmetic mean can hide critical performance details. The average is highly sensitive to **outliers**; for example, a single request delayed by a network error can skew the mean, making the system appear much slower than it is for the vast majority of users. 

Furthermore, the average fails to account for **tail latencies** at high percentiles (like p95 or p99), which often represent the experience of a company's **most valuable customers** who have the most data on their accounts. To gain a clearer understanding of system health, it is better to use **percentiles**, which distinguish between the typical user experience (the median) and extreme cases that may indicate underlying technical issues."""
        },
        "metadata": {
            "external_id": "id_8",
        },
    },
    {
        "inputs": {
            "question": "When a system batches online queries concurrently, what is the typical trade-off between latency and throughput?",
            "document_id": "1ed41248-cf8a-4231-8a33-415191dc9c31",
            "user_id": "f9937aab-6c97-4c3e-a6f8-38f4a1676200",
        },
        "outputs": {
            "answer": """When a system batches online queries concurrently, **higher latency often enables higher throughput**. While traditional sequential systems see throughput drop as latency increases, distributed systems use batching to process multiple requests at once, allowing them to handle significantly more queries per second even if individual response times are slower.

However, batching online queries introduces a specific **latency penalty** because the system must **wait for a sufficient number of queries to arrive** to form a batch before processing can begin. This creates a fundamental conflict in production:

*   **To maximize throughput:** Systems use aggressive batching to saturate hardware, which increases the time users wait for a response.
*   **To minimize latency:** Systems must reduce batch sizes or process queries individually, which leads to **underutilized hardware** and a higher cost per query.

In production, teams typically prioritize **low latency** because even small delays (e.g., 100 ms) can significantly hurt user conversion rates, whereas research settings often prioritize **high throughput** through aggressive batching."""
        },
        "metadata": {
            "external_id": "id_9",
        },
    },
    {
        "inputs": {
            "question": "What characteristic makes a problem especially suitable for an ML solution, according to the 'When to Use Machine Learning' section?",
            "document_id": "1ed41248-cf8a-4231-8a33-415191dc9c31",
            "user_id": "f9937aab-6c97-4c3e-a6f8-38f4a1676200",
        },
        "outputs": {
            "answer": """According to the **"When to Use Machine Learning"** section, a problem is especially suitable for an ML solution if it possesses the following four characteristics:

*   **It is repetitive:** Most ML algorithms require many examples to learn a pattern; therefore, a task where patterns are repeated multiple times makes it easier for machines to learn.
*   **The cost of wrong predictions is cheap:** ML is ideal when a mistake is forgiving (like a bad movie recommendation) or when the average benefit of correct predictions outweighs the cost of occasional errors.
*   **It is at scale:** Because ML requires significant up-front investment in data and infrastructure, it is most effective for tasks requiring a large volume of predictions, such as sorting millions of emails.
*   **The patterns are constantly changing:** ML is superior to handcoded rules for tasks where patterns evolve (like spam detection) because models can be updated with new data without needing to manually rediscover the changing rules."""
        },
        "metadata": {
            "external_id": "id_10",
        },
    },
    {
        "inputs": {
            "question": "In the restaurant recommendation app example, why did the ML platform team want to delay model updates?",
            "document_id": "1ed41248-cf8a-4231-8a33-415191dc9c31",
            "user_id": "f9937aab-6c97-4c3e-a6f8-38f4a1676200",
        },
        "outputs": {
            "answer": """In the restaurant recommendation app example, the **ML platform team** wanted to delay model updates so they could **prioritize improving the ML platform's infrastructure**. As user traffic grew, the team faced significant **scaling problems** with the existing system that were severe enough to cause them to be "woken up in the middle of the night" to address technical issues."""
        },
        "metadata": {
            "external_id": "id_11",
        },
    },
    {
        "inputs": {
            "question": "Which percentile is commonly used to discover outliers and specify strict performance requirements for system latency?",
            "document_id": "1ed41248-cf8a-4231-8a33-415191dc9c31",
            "user_id": "f9937aab-6c97-4c3e-a6f8-38f4a1676200",
        },
        "outputs": {
            "answer": """To discover outliers and specify strict performance requirements for system latency, high percentiles such as **p90, p95, p99, and p99.9** are commonly used. 

*   **Discovering Outliers:** Typically, **p90, p95, and p99** are used to identify outliers that may indicate system symptoms or network errors. For example, a p90 value significantly higher than the median can reveal a specific request causing a massive delay.
*   **Strict Performance Requirements:** Product managers and engineers often use the **90th percentile (p90)** or the **99.9th percentile (p99.9)** to set performance benchmarks in Service Level Agreements (SLAs). A system might be required to keep its p99.9 latency below a certain threshold to ensure reliability.

Focusing on these high percentiles is vital because they often represent the experience of the company's **most valuable customers**, who typically have the most data associated with their accounts."""
        },
        "metadata": {
            "external_id": "id_12",
        },
    },
    {
        "inputs": {
            "question": "What is 'zero-shot learning' as described in the text?",
            "document_id": "1ed41248-cf8a-4231-8a33-415191dc9c31",
            "user_id": "f9937aab-6c97-4c3e-a6f8-38f4a1676200",
        },
        "outputs": {
            "answer": """**Zero-shot learning** (also known as **zero-data learning**) is a scenario where a machine learning system can make accurate predictions for a specific task **without having been trained on any data for that task**. Instead, the system leverages knowledge from being previously trained on **other related tasks**, allowing a base model to be used directly for a downstream task without requiring new ground truth labels."""
        },
        "metadata": {
            "external_id": "id_13",
        },
    },
    {
        "inputs": {
            "question": "What is the key difference between research data and production data?",
            "document_id": "1ed41248-cf8a-4231-8a33-415191dc9c31",
            "user_id": "f9937aab-6c97-4c3e-a6f8-38f4a1676200",
        },
        "outputs": {
            "answer": """The key difference is that research data is typically **static and historical**, while production data is **messy and constantly shifting**. 

Specific distinctions include:
*   **Quality and Format:** Research datasets are usually **clean and well-formatted** to allow for benchmarking, whereas production data is often **noisy, unstructured, and biased**.
*   **Availability:** Research relies on **fixed, finite sets** stored in databases, while production involves data that is **constantly generated** by users and systems in real-time.
*   **Labels:** Research data typically has known ground truths, but labels in production can be **sparse, imbalanced, incorrect**, or subject to significant delays.

Ultimately, while research data is stationary, production data is **neither finite nor stationary**, which frequently leads to performance degradation over time."""
        },
        "metadata": {
            "external_id": "id_14",
        },
    },
    {
        "inputs": {
            "question": "According to the European Union's GDPR, under what conditions can personal data be processed without explicit user consent?",
            "document_id": "1ed41248-cf8a-4231-8a33-415191dc9c31",
            "user_id": "f9937aab-6c97-4c3e-a6f8-38f4a1676200",
        },
        "outputs": {
            "answer": """The provided sources do not explicitly list the legal conditions under which the GDPR allows personal data to be processed without explicit user consent (such as "legitimate interests" or "contractual necessity"). """
        },
        "metadata": {
            "external_id": "id_15",
        },
    },
    {
        "inputs": {
            "question": "In the official Kubernetes documentation, what is the difference between a Deployment and a StatefulSet?",
            "document_id": "1ed41248-cf8a-4231-8a33-415191dc9c31",
            "user_id": "f9937aab-6c97-4c3e-a6f8-38f4a1676200",
        },
        "outputs": {
            "answer": """The provided sources, which focus on designing machine learning systems, do not contain information regarding the specific differences between a Deployment and a StatefulSet as defined in the official Kubernetes documentation."""
        },
        "metadata": {
            "external_id": "id_16",
        },
    },
    {
        "inputs": {
            "question": "What were the main terms agreed upon in the Treaty of Tordesillas, and which countries were involved?",
            "document_id": "1ed41248-cf8a-4231-8a33-415191dc9c31",
            "user_id": "f9937aab-6c97-4c3e-a6f8-38f4a1676200",
        },
        "outputs": {
            "answer": """The provided sources, which focus on the technical and operational aspects of designing machine learning systems, do not contain any information regarding the Treaty of Tordesillas."""
        },
        "metadata": {
            "external_id": "id_17",
        },
    },
    {
        "inputs": {
            "question": "In the original Transformer paper “Attention Is All You Need,” what role does positional encoding play in the model?",
            "document_id": "1ed41248-cf8a-4231-8a33-415191dc9c31",
            "user_id": "f9937aab-6c97-4c3e-a6f8-38f4a1676200",
        },
        "outputs": {
            "answer": """In the original Transformer paper, "Attention Is All You Need" (Vaswani et al. 2017), **positional encoding** is used to provide the model with information about the **relative or absolute position of tokens** in a sequence.

Because the Transformer architecture processes words in **parallel** rather than sequentially (like recurrent neural networks), it has no inherent sense of the order of words. Positional encoding explicitly inputs this information so the model can distinguish between different word orders, such as "a dog bites a child" versus "a child bites a dog".

In the original paper, these encodings are **fixed** (predefined) using **sine and cosine functions** of different frequencies:
*   **Sine functions** are used for even-indexed elements in the embedding vector.
*   **Cosine functions** are used for odd-indexed elements.

These positional encodings are the same size as the word embeddings so that the two vectors can be **summed together** and passed into the model."""
        },
        "metadata": {
            "external_id": "id_18",
        },
    },
    {
        "inputs": {
            "question": "According to the official OpenAI API documentation, what are the rate limits and how are they enforced?",
            "document_id": "1ed41248-cf8a-4231-8a33-415191dc9c31",
            "user_id": "f9937aab-6c97-4c3e-a6f8-38f4a1676200",
        },
        "outputs": {
            "answer": """The provided sources do not contain information regarding the official OpenAI API documentation, specifically its rate limits or enforcement mechanisms. The sources primarily focus on the technical and operational aspects of designing and maintaining production-ready machine learning systems, mentioning OpenAI only in the context of the training data and parameters of its GPT models"""
        },
        "metadata": {
            "external_id": "id_19",
        },
    },
    {
        "inputs": {
            "question": "What is the capital of South Africa?",
            "document_id": "1ed41248-cf8a-4231-8a33-415191dc9c31",
            "user_id": "f9937aab-6c97-4c3e-a6f8-38f4a1676200",
        },
        "outputs": {
            "answer": """The provided sources, which focus on the technical and operational aspects of designing machine learning systems, do not contain information regarding the capital of South Africa."""
        },
        "metadata": {
            "external_id": "id_20",
        },
    },
]
