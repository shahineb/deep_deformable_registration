import os
import sys
import logging
import verboselogs
import pandas as pd

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)
from src.generators import luna_generator as gen
from src.training.config_file import ConfigFile
import src.metrics as metrics


class LunaTester:
    """Utility class to assess a model performances wrt a given set of metrics

    Attributes:
        reg_metrics (dict): dictionnary of default metrics used for image registration
        test_ids_filename (str): filename for list of test ids used
        test_scores_filename (str): filename for dumped score file

    """

    reg_metrics = {'mse': metrics.mse,
                   'cross_correlation': metrics.cross_correlation}

    seg_metrics = {'dice': metrics.dice_score,
                   'haussdorf': metrics.haussdorf_distance}

    test_ids_filename = "test_ids.csv"
    train_scores_filename = "train_scores.csv"
    val_scores_filename = "val_scores.csv"
    test_scores_filename = "test_scores.csv"

    def __init__(self,
                 model=None,
                 device=None,
                 config_path=None,
                 weights_path=None,
                 reg_metric_dict=None,
                 seg_metric_dict=None,
                 verbose=1):
        """
        Args:
            model (keras.model): model architecture
            metric_dict (dict): metric dictionnary following LunaTester.reg_metrics format
            config_path (str): path to serialized config file following src.training.ConfigFile
            weights_path (str): path to model weights (optional)
            use_segmentation (boolean): if true trains with segmentation data
            verbose (int): {0, 1}
        """
        self._model = model
        self._device = device
        self._weights_path = weights_path
        if self._weights_path:
            self._model.load_weights(self._weights_path)
        self._reg_metric_dict = reg_metric_dict
        self._seg_metric_dict = seg_metric_dict
        self._metric_dict = reg_metric_dict
        if self._metric_dict and self._seg_metric_dict:
            self._metric_dict.update(self._seg_metric_dict)
        self._config = ConfigFile(session_name="")
        if config_path:
            self._config.load(config_path)
        self._verbose = verbose
        self._logger = verboselogs.VerboseLogger('verbose-demo')
        self._logger.addHandler(logging.StreamHandler())
        if verbose:
            self._logger.setLevel(verbose)

    def get_reg_metric_dict(self):
        return self._reg_metric_dict

    def get_seg_metric_dict(self):
        return self._seg_metric_dict

    def get_metric_dict(self):
        return self._metric_dict

    def get_metric(self, metric_name):
        try:
            return self._metric_dict[metric_name]
        except KeyError:
            raise KeyError(f"Unkown <{metric_name}> metric")

    @staticmethod
    def _wrap_metric(pred, ground_truth, metric):
        """Wrapper to evaluate metric on prediction and ground_truth

        Args:
            pred (np.ndarray)
            ground_truth (np.ndarray)
            metric (function)
        """
        return metric(pred, ground_truth)

    def evaluate_sample(self, sample, use_affine, use_segmentation):
        """Computes model prediction based on reference and moving images

        Args:
            sample (list[np.ndarray]): generator output formatted as
                - ([src_scan, tgt_scan], [tgt_scan, identity_flow]) default
                - ([src_scan, tgt_scan], [tgt_scan, identity_flow, identity_affine]) if affine
                - ([src_scan, tgt_scan, src_seg], [tgt_scan, identity_flow, tgt_seg]) if segmentation
            src_scan (np.ndarray)
            tgt_scan (np.ndarray)
            use_segmentation (bool): if true, computes scores on segmentation registration
        """
        if use_segmentation and use_affine:
            [deformed, flow, affine_flow, deformed_seg] = self._model.predict(sample[0])
        elif use_segmentation:
            [deformed, flow, deformed_seg] = self._model.predict(sample[0])
        elif use_affine:
            [deformed, flow, affine_flow] = self._model.predict(sample[0])
        else:
            [deformed, flow] = self._model.predict(sample)
        scores = dict.fromkeys(self._metric_dict.keys(), None)
        for metric_name, metric in self._reg_metric_dict.items():
            scores.update({metric_name: [LunaTester._wrap_metric(sample[0][1].squeeze(), deformed.squeeze(), metric)]})
        if use_segmentation:
            if use_affine:
                for metric_name, metric in self._seg_metric_dict.items():
                    scores.update({metric_name: [LunaTester._wrap_metric(sample[1][3].squeeze(), deformed_seg.squeeze(), metric)]})
            else:
                for metric_name, metric in self._seg_metric_dict.items():
                    scores.update({metric_name: [LunaTester._wrap_metric(sample[1][2].squeeze(), deformed_seg.squeeze(), metric)]})
        return scores

    def evaluate(self, scan_ids, filename, generator="luna", use_affine=False):
        """Evaluates model performances on a testing set for all the specified
        metrics

        Args:
            scan_ids (list): list of scans ids

        Returns:
            scores_df (pd.DataFrame): scores dataframe
        """
        self._logger.verbose(f"Number of testing scans : {len(scan_ids)}\n")
        use_segmentation = generator in ["luna_seg", "atlas_seg"]
        pd.DataFrame(scan_ids).to_csv(os.path.join(self._config.session_dir, LunaTester.test_ids_filename), index=False)

        (width, height, depth) = self._config.input_shape
        if generator in ["luna", "luna_seg"]:
            eval_gen = gen.catalog[generator](scan_ids, width, height, depth, loop=False, shuffle=True, use_affine=use_affine)
        elif generator in ["atlas", "atlas_seg"]:
            if not self._config.atlas_id:
                raise RuntimeError("Must specify an atlas id if using atlas registration")
            eval_gen = gen.catalog[generator](self._config.atlas_id, scan_ids, width, height, depth, loop=False, shuffle=True, use_affine=use_affine)
        else:
            raise UnboundLocalError(f"Unkown specified generator, specify within {gen.generator_catalog.keys()}")
        scores = dict.fromkeys(self._metric_dict.keys(), [])
        i = 0
        self._logger.verbose(f"********** Beginning evaluation **********\n")
        try:
            while True:
                sample = next(eval_gen, use_affine, use_segmentation)
                sample_score = self.evaluate_sample(sample)
                scores.update({k: scores[k] + sample_score[k] for k in scores.keys()})

                if i % (len(scan_ids) // 100 + 1) == 0:
                    # TODO : find clean way to log (tqdm for while True ?)
                    self._logger.verbose(f"Evaluated {i}/{len(scan_ids)} scans\n")
                    pd.DataFrame.from_dict(scores).to_csv(os.path.join(self._config.session_dir, LunaTester.test_scores_filename))
                i += 1
        except StopIteration:
            self._logger.verbose(f"Evaluation completed !\n")
            scores_df = pd.DataFrame.from_dict(scores)
            scores_df.to_csv(os.path.join(self._config.session_dir, ConfigFile.scores_dirname, filename))
            return scores_df
