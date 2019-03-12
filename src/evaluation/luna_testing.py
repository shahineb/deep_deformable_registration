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

    test_ids_filename = "test_ids.csv"
    test_scores_filename = "test_scores.csv"

    def __init__(self, 
                 model=None, 
                 metric_dict=None,
                 config_path=None,
                 weights_path=None,
                 use_segmentation=False,
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
        self.model_ = model
        self.weights_path_ = weights_path
        if self.weights_path_:
            self.model_.load_weights(self.weights_path_)
        self.metric_dict_ = metric_dict
        self.config = ConfigFile(session_name="")
        if config_path:
            self.config.load(config_path)
        self.use_segmentation_ = use_segmentation
        self.verbose_ = verbose
        self.logger = verboselogs.VerboseLogger('verbose-demo')
        self.logger.addHandler(logging.StreamHandler())
        if verbose:
            self.logger.setLevel(verbose)

    def get_metric_dict(self):
        return self.metric_dict_

    def get_metric(self, metric_name):
        return self.metric_dict_[metric_name]

    @staticmethod
    def _score_sample(pred, ground_truth, metric):
        """Evaluates metric on prediction and ground_truth

        Args:
            pred (np.ndarray)
            ground_truth (np.ndarray)
            metric (function)
        """
        return metric(pred, ground_truth)

    def score_sample(self, src_scan, tgt_scan):
        """Computes model prediction based on reference and moving images

        Args:
            src_scan (np.ndarray)
            tgt_scan (np.ndarray)
        """
        # TODO : extend to segmentation
        output = self.model_.predict([src_scan, tgt_scan])
        scores = dict.fromkeys(self.metric_dict_.keys(), None)
        for metric_name, metric in self.metric_dict_.items():
            scores.update({metric_name: [LunaTester._score_sample(tgt_scan, output[0], metric)]})
        return scores

    def evaluate(self, test_ids, generator="luna", atlas_id=None, use_affine=False):
        """Evaluates model performances on a testing set for all the specified
        metrics

        Args:
            test_ids (list): list of scans ids

        Returns:
            scores_df (pd.DataFrame): scores dataframe
        """
        self.logger.verbose(f"Number of testing scans : {len(test_ids)}\n")
        pd.DataFrame(test_ids).to_csv(os.path.join(self.config.session_dir, LunaTester.test_ids_filename), index=False)

        (width, height, depth) = self.config.input_shape
        if generator == "luna":
            test_gen = gen.scan_generator(test_ids, width, height, depth, loop=False, shuffle=True, use_affine=use_affine)
        elif generator == "segmentation":
            test_gen = gen.scan_and_seg_generator(test_ids, width, height, depth, loop=False, shuffle=True, use_affine=use_affine)
        elif generator == "atlas":
            if not atlas_id:
                raise RuntimeError("Must specify an atlas id if using atlas registration")
            test_gen = gen.atlas_generator(atlas_id, test_ids, width, height, depth, loop=False, shuffle=True, use_affine=use_affine)
        elif generator == "atlas_seg":
            if not atlas_id:
                raise RuntimeError("Must specify an atlas id if using atlas registration")
            test_gen = gen.atlas_seg_generator(atlas_id, test_ids, width, height, depth, loop=False, shuffle=True, use_affine=use_affine)
        else:
            raise UnboundLocalError("Unkown specified generator")
        scores = dict.fromkeys(self.metric_dict_.keys(), [])
        i = 0
        self.logger.verbose(f"********** Beginning evaluation **********\n")
        try:
            while True:
                if self.use_segmentation_:
                    ([src_scan, tgt_scan, _], __) = next(test_gen)
                    del _, __
                else:
                    ([src_scan, tgt_scan], _) = next(test_gen)
                sample_score = self.score_sample(src_scan, tgt_scan)
                scores.update({k: scores[k] + sample_score[k] for k in scores.keys()})

                if i % (len(test_ids) // 100 + 1) == 0:
                    self.logger.verbose(f"Evaluated {i}/{len(test_ids)} scans\n")
                    pd.DataFrame.from_dict(scores).to_csv(os.path.join(self.config.session_dir, LunaTester.test_scores_filename))
                i += 1
        except StopIteration:
            self.logger.verbose(f"Evaluation completed !\n")
            scores_df = pd.DataFrame.from_dict(scores)
            scores_df.to_csv(os.path.join(self.config.session_dir, LunaTester.test_scores_filename))
            return scores_df
