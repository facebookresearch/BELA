from pathlib import Path
from itertools import product

from tqdm import tqdm
import numpy as np

from bela.evaluation.model_eval import ModelEval, load_file
from bela.utils.prediction_utils import get_predictions_using_windows


if __name__ == "__main__":
    # Model finetuned on TACKBP
    checkpoint_path = "/data/home/louismartin/dev/BELA/multirun/2023-02-17/00-08-45/0/lightning_logs/version_0/checkpoints/checkpoint_best-v2.ckpt"
    model_eval = ModelEval(checkpoint_path, config_name="joint_el_mel_new_index")

    results = {}
    print(f"{model_eval.checkpoint_path=}")
    for md_threshold, el_threshold in tqdm(list(product(np.arange(0, 0.6, 0.2), repeat=2))):
        model_eval.task.md_threshold = md_threshold
        model_eval.task.el_threshold = el_threshold
        print(f"{model_eval.task.md_threshold=}")
        print(f"{model_eval.task.el_threshold=}")
        test_data_path = "/fsx/louismartin/bela/retrieved_from_aws_backup/ndecao/TACKBP2015/train_bela_format_all_languages.jsonl",
        print(f"Processing {test_data_path}")
        test_data = load_file(test_data_path)
        predictions = get_predictions_using_windows(model_eval, test_data, window_length=1024)
        (f1, precision, recall), (f1_boe, precision_boe, recall_boe) = ModelEval.compute_scores(test_data, predictions)
        print(f"F1 = {f1:.4f}, precision = {precision:.4f}, recall = {recall:.4f}")
        print(f"F1 boe = {f1_boe:.4f}, precision = {precision_boe:.4f}, recall = {recall_boe:.4f}")
        results[(md_threshold, el_threshold)] = (f1, precision, recall), (f1_boe, precision_boe, recall_boe)
    print(sorted(results.items(), key=lambda x: x[1][0][0], reverse=True))
    pickle_path = Path.home() / "tmp/grid_search_thresholds.pkl"
