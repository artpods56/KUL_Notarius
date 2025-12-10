import numpy as np
import evaluate
from transformers.models.tapas.modeling_tf_tapas import n

metric = evaluate.load("seqeval")


def build_compute_metrics(label_list, return_entity_level_metrics=True):
    def compute_metrics(p) -> dict[str, float]:
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [label_list[pred] for (pred, lab) in zip(prediction, label) if lab != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[lab] for (pred, lab) in zip(prediction, label) if lab != -100]
            for prediction, label in zip(predictions, labels)
        ]

        metrics = metric.compute(predictions=true_predictions, references=true_labels)

        if metrics is None:
            raise ValueError(f"Couldn't compute metrics for label_list={label_list}")

        if return_entity_level_metrics:
            final_metrics: dict[str, float] = {}
            for key, value in metrics.items():
                if isinstance(value, dict):
                    for sub_key, sub_val in value.items():
                        final_metrics[f"eval_{sub_key.lower()}_{key.capitalize()}"] = (
                            sub_val
                        )
                else:
                    final_metrics[f"eval_{key}"] = value
            return final_metrics
        else:
            return {
                "eval_precision": metrics["overall_precision"],
                "eval_recall": metrics["overall_recall"],
                "eval_f1": metrics["overall_f1"],
                "eval_accuracy": metrics["overall_accuracy"],
            }

    return compute_metrics
