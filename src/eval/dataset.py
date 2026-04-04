"""Hardcoded evaluation dataset and LangSmith upload helpers.

Contains placeholder examples that must be replaced with real values
before running end-to-end evaluation.
"""

from langsmith import Client

DATASET_NAME = "sanctuary"


examples = [
    {
        "inputs": {
            "question": "How do I handle data distribution shifts in a live environment?",
            "document_id": "1ed41248-cf8a-4231-8a33-415191dc9c31",
            "user_id": "f9937aab-6c97-4c3e-a6f8-38f4a1676200",
        },
        "outputs": {
            "answer": """**Detecting drift** is your first priority. In a live environment, the statistical properties of your incoming data will inevitably diverge from what your model was trained on — this is called distribution shift or concept drift. You catch it by continuously monitoring key signals: use statistical tests like the KS test or Population Stability Index (PSI) on your input features, watch your model's prediction confidence distribution, and track output label proportions over time. Tools like Evidently AI, Alibi Detect, or WhyLogs make this operationally manageable. The critical insight here is to monitor inputs and outputs *before* accuracy degrades — by the time your metrics drop visibly, you've already been serving bad predictions for days.

**Once drift is detected, your remediation strategy depends on what shifted.** If only the input distribution changed (covariate shift), importance weighting lets you retrain on your existing labeled data while correcting for the mismatch. If the underlying relationship between features and labels changed (concept drift), you need fresh labeled data and a retrained model. The most common pattern is sliding-window retraining — continuously retraining on a recent time window of data, optionally weighting newer samples more heavily. For high-frequency data streams where batch retraining is too slow, online learning frameworks like River let the model update incrementally with each new sample.

**Architecturally, build drift handling in from the start.** Log every prediction alongside its input features and timestamp so you can do retrospective analysis. Store ground truth labels as they arrive and join them back to predictions. Use shadow/challenger models running in parallel so a freshly retrained model can be validated on live traffic before promotion. Implement canary deployments to safely roll out updated models to a traffic slice first. Finally, version your datasets alongside your models — knowing exactly which data produced which model behavior is essential for debugging when things go wrong in production.""",
        },
    },
    {
        "inputs": {
            "question": "How do I ensure my model performs as well in production as it did during training?",
            "document_id": "1ed41248-cf8a-4231-8a33-415191dc9c31",
            "user_id": "f9937aab-6c97-4c3e-a6f8-38f4a1676200",
        },
        "outputs": {
            "answer": """The gap between training performance and production performance almost always comes down to **data leakage, distribution mismatch, or evaluation methodology errors** made before deployment. The fix starts at the very beginning of your pipeline, not at deployment time. First, ensure your validation strategy honestly reflects production conditions: use time-based splits instead of random splits if your data is temporal, hold out data from entirely unseen user segments or geographies, and never let any future information leak into your training features. A model that scores 95% AUC in a leaky offline evaluation might drop to 70% in production, and the gap will be invisible until it's too late. Invest heavily in building a validation set that is a faithful replica of what the model will actually see.

**Closing the training-serving skew** is the second major lever. This happens when the features your model sees during training are computed differently than the features served at inference time — a surprisingly common and painful problem. The solution is to use a shared feature store so that the exact same transformation code runs in both training and serving pipelines. Shadow your production environment during training: replicate the same data types, null rates, encoding schemes, and preprocessing steps. Log a sample of live inference requests regularly and compare their feature distributions against your training data using PSI or KS tests. Any divergence between the two is silent model degradation waiting to happen.

**Finally, treat production as a continuous experiment, not a finish line.** Implement structured A/B testing and canary deployments so every model update is validated on real traffic before full rollout. Collect ground truth labels from production as fast as your business allows — user clicks, conversions, outcomes — and feed them back into your evaluation loop. Set up automated retraining triggers when drift is detected rather than retraining on a fixed calendar schedule. Monitor not just accuracy but business metrics, since a model can look statistically healthy while quietly optimizing the wrong objective. The models that hold up in production are the ones backed by a monitoring and feedback infrastructure that treats deployment as the beginning of the model's lifecycle, not the end.""",
        },
    },
    {
        "inputs": {
            "question": "How do I detect and fix a 'silent failure' once the model is deployed?",
            "document_id": "1ed41248-cf8a-4231-8a33-415191dc9c31",
            "user_id": "f9937aab-6c97-4c3e-a6f8-38f4a1676200",
        },
        "outputs": {
            "answer": """**Silent failures are dangerous precisely because no alarm goes off** — the model continues serving predictions confidently while its quality degrades invisibly. They typically stem from three root causes: upstream data pipeline changes that alter feature distributions without breaking schemas, concept drift where the real-world relationship between inputs and outputs has shifted, or a business metric decoupling where the model's statistical performance looks fine but it's quietly optimizing the wrong thing. The first line of defense is monitoring inputs, not just outputs. Log every inference request with its full feature vector and timestamp, then run continuous statistical checks — PSI on numeric features, chi-squared on categoricals — comparing live distributions against your training baseline. If a feature that was never null during training suddenly shows 30% nulls in production, your model is operating in territory it was never designed for, and accuracy metrics alone will never surface that.

**Once you suspect a silent failure, your debugging workflow should be systematic.** Start by slicing your prediction logs by time, user segment, geography, or any other dimension you can think of — aggregate metrics often mask severe degradation in subgroups. Compare your live prediction confidence distribution against what it looked like at deployment; a model that was previously sharp but is now producing flat, uncertain probability distributions has lost its signal. Next, join your predictions back to any available ground truth labels, even partial or delayed ones, and compute metrics on that joined subset. If you can't get labels quickly, use proxy signals — engagement, downstream conversions, complaint rates — as early warning indicators. Build a confusion matrix over time and look for asymmetric degradation, where one class is being systematically misclassified while overall accuracy stays artificially inflated by class imbalance.

**Fixing a silent failure requires both an immediate response and a structural one.** In the short term, roll back to the last known good model version using your canary or blue-green deployment infrastructure, buying yourself time to investigate without continued damage. Then trace the failure back through your pipeline: check whether any upstream data source changed its schema, encoding, or null behavior around the time degradation began. Re-examine your feature engineering code for any step that behaves differently at scale or under production data conditions versus your training environment. For the structural fix, retrain on recent data that reflects the current distribution, validate it against a holdout set drawn from the same recent window, and deploy it through a shadow-mode comparison against the incumbent before promotion. Long term, the real fix is building the observability infrastructure that catches the next silent failure before it becomes a crisis — automated drift alerts, prediction distribution monitoring, and a fast ground truth feedback loop are not optional luxuries, they are the foundation of a trustworthy production ML system."""
        },
    },
]


def ensure_dataset(client: Client | None = None) -> str:
    """Create the evaluation dataset in LangSmith if it does not already exist.

    Uploads hardcoded examples idempotently — skips creation when a dataset
    with the same name is already present.

    Args:
        client: Optional LangSmith client. Created automatically if not provided.

    Returns:
        The dataset name (for use with ``langsmith.evaluate()``).
    """
    if client is None:
        client = Client()

    if not client.has_dataset(dataset_name=DATASET_NAME):
        dataset = client.create_dataset(dataset_name=DATASET_NAME)
        client.create_examples(
            inputs=[ex["inputs"] for ex in examples],
            outputs=[ex["outputs"] for ex in examples],
            dataset_id=dataset.id,
        )

    return DATASET_NAME
