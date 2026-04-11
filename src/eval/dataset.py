"""LangSmith dataset upload helpers.

Syncs evaluation examples to a LangSmith dataset, handling
creation, updates, and deletions via content-hash diffing.
"""

import hashlib
import json
from typing import Any

from langsmith import Client

from src.eval.examples import examples

DATASET_NAME = "sanctuary"


def get_content_hash(inputs: dict[str, Any], outputs: dict[str, Any]) -> str:
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
