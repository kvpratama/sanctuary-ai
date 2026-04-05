"""Hardcoded evaluation dataset and LangSmith upload helpers.

Contains placeholder examples that must be replaced with real values
before running end-to-end evaluation.
"""

import hashlib
import json

from langsmith import Client

DATASET_NAME = "sanctuary"


examples: list[dict] = [
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

Once a failure is detected, you must first **diagnose the root cause**, as a large percentage of perceived data shifts are actually **internal human errors** like pipeline bugs, missing values, or inconsistent feature extraction between training and inference. It is vital to rule out these operational issues before assuming a fundamental change in the environment has occurred. If the failure is indeed caused by a **data distribution shift**, such as covariate shift or concept drift, you must adapt the model to the new target distribution. **Statistical methods**, including two-sample hypothesis tests like the Kolmogorov–Smirnov test, can help confirm if the difference between production data and training data is statistically significant enough to merit intervention.

To fix these failures and increase future reliability, the most common industry approach is **retraining the model** with fresh labeled data from the target distribution, utilizing either **stateless retraining** (from scratch) or **stateful training (fine-tuning)** to save compute resources. Beyond reactive retraining, you can **design for robustness** by choosing stable features—for instance, bucketing rapidly changing raw rankings into broader categories. Implementing **smooth failing** with a backup system, such as a simple heuristic or a set of precomputed predictions, ensures the system remains useful if the main model produces suspicious outputs. Finally, to correct **degenerate feedback loops** that bias future training data, you can introduce **randomization** to explore new items or use positional features to decouple a model's prediction from its rank on a screen."""
        },
        "metadata": {
            "external_id": "id_3",
        },
    },
]


def get_content_hash(inputs: dict, outputs: dict) -> str:
    """Generate a unique SHA-256 hash from combined inputs and outputs.

    Args:
        inputs: Dictionary of input values to include in the hash.
        outputs: Dictionary of output values to include in the hash.

    Returns:
        A hex-encoded SHA-256 hash string unique to the combined content.
    """
    # 1. Combine inputs and outputs into a predictable string
    combined = json.dumps({"i": inputs, "o": outputs}, sort_keys=True)
    # 2. Create a SHA-256 hash (or MD5 for slightly faster, less strict usage)
    return hashlib.sha256(combined.encode()).hexdigest()


def ensure_dataset(client: Client | None = None) -> str:
    """Create or sync the evaluation dataset in LangSmith.

    Creates the dataset with all hardcoded examples if it does not exist.
    If it already exists, performs a diff-based sync so that any changes
    to the ``examples`` list (additions, deletions, modifications) are
    reflected in LangSmith.

    Args:
        client: Optional LangSmith client. Created automatically if not provided.

    Returns:
        The dataset name (for use with ``langsmith.evaluate()``).
    """
    if client is None:
        client = Client()

    # 1. Ensure dataset exists
    try:
        dataset = client.read_dataset(dataset_name=DATASET_NAME)
    except Exception:
        dataset = client.create_dataset(dataset_name=DATASET_NAME)

    # 2. Get existing examples to compare
    # Note: If your dataset is huge, you may need to paginate or filter
    existing_examples = list(client.list_examples(dataset_id=dataset.id))
    existing_map = {
        ex.metadata.get("external_id"): ex for ex in existing_examples if ex.metadata
    }

    existed, updated, new_examples, deleted = [], [], [], []

    # 3. Iterate through your local examples to create/update
    local_external_ids: set[str | None] = set()
    for ex in examples:
        external_id = ex["metadata"]["external_id"]
        local_external_ids.add(external_id)
        current_hash = get_content_hash(ex["inputs"], ex["outputs"])

        # Merge your external ID with the hash for tracking
        metadata = ex.get("metadata", {}).copy()
        metadata["content_hash"] = current_hash

        if external_id in existing_map:
            # Check if existing content matches current hash

            existing_ex = existing_map[external_id]
            if (
                existing_ex.metadata
                and existing_ex.metadata.get("content_hash") != current_hash
            ):
                client.update_example(
                    example_id=existing_ex.id,
                    inputs=ex["inputs"],
                    outputs=ex["outputs"],
                    metadata=metadata,
                )
                updated.append(ex)
            else:
                existed.append(ex)
        else:
            # New example - upload it
            client.create_example(
                dataset_id=dataset.id,
                inputs=ex["inputs"],
                outputs=ex["outputs"],
                metadata=metadata,
            )
            new_examples.append(ex)

    # 4. Delete remote examples that no longer exist locally
    for external_id, existing_ex in existing_map.items():
        if external_id not in local_external_ids:
            client.delete_example(example_id=existing_ex.id)
            deleted.append(existing_ex)

    print(
        f"Updated {len(updated)} examples, \n"
        f"created {len(new_examples)} new examples, \n"
        f"deleted {len(deleted)} examples, and \n"
        f"found {len(existed)} existing examples.\n"
    )

    return DATASET_NAME
