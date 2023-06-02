# Copyright 2023 (c) OpenAI.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training and evaluation"""

from jcm import train
from jcm import evaluate
from jcm import metrics
import logging
import os
import blobfile
import wandb
from absl import flags, app
from ml_collections.config_flags import config_flags

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=True
)
flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_enum(
    "mode",
    None,
    ["train", "eval", "metrics"],
    "Running mode: train or eval or metrics",
)
flags.DEFINE_string(
    "eval_folder", "eval", "The folder name for storing evaluation results"
)
flags.DEFINE_integer("num_gpus", 8, "Number of GPUs to use.")
flags.mark_flags_as_required(["workdir", "config", "mode"])


def main(argv):
    if FLAGS.mode == "train":
        wandb.login()
        wandb.init(
            project=os.path.basename(FLAGS.workdir),
            name=os.path.basename(FLAGS.workdir),
            config=FLAGS.config.to_dict(),
        )

        # Create the working directory
        blobfile.makedirs(FLAGS.workdir)
        formatter = logging.Formatter(
            "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
        )
        logger = logging.getLogger()
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel("INFO")
        # Run the training pipeline
        train.train(FLAGS.config, FLAGS.workdir)
    elif FLAGS.mode == "eval":
        # Run the evaluation pipeline
        evaluate.evaluate(
            FLAGS.config,
            FLAGS.workdir,
            FLAGS.eval_folder,
        )
    elif FLAGS.mode == "metrics":
        # Compute the metrics
        metrics.compute_metrics(
            FLAGS.config,
            FLAGS.workdir,
            FLAGS.eval_folder,
        )
    else:
        raise ValueError(f"Mode {FLAGS.mode} not recognized.")


if __name__ == "__main__":
    app.run(main)
