# 🔮 Benchmarking SOT Feature Transforms for Biomedical Few-Shot Learning Tasks

[SOT](https://arxiv.org/abs/2204.03065) is a parameter-less feature transform module, grounded in probabilistic interpretations, that promises to improve the clustering of an arbitrary feature matrix in an embedding space and thereby improve the performance of downstream tasks relying on discriminative embeddings. The SOT feature transform has shown promising results in improving few-shot learning methods.

The primary objective of this study is to explore the effectiveness of the SOT feature transform in the context of few-shot learning tasks within the biomedical domain. Our approach involves establishing a comprehensive benchmarking suite to compare a range of few-shot learning algorithms, including [Baseline](https://arxiv.org/abs/1904.04232), [MAML](https://arxiv.org/abs/1703.03400), [ProtoNet](https://arxiv.org/abs/1703.05175) and [MatchingNet](https://arxiv.org/abs/1606.04080) on two few-shot classification tasks which are sourced from the Tabula Muris cell type annotation dataset and the SwissProt protein function prediction dataset.

## ⚙️ Environment Setup

This project was developed in Python `3.10`. Any minor version of Python 3 should work, but we recommend using the latest release. You can install the dependencies of the project using your favourite tool e.g. `pip` from `requirements.txt` or `conda` from `environment.yml`.

## 💡 Usage

The project contains a single entry point `run.py` which is used to run a single experiment. Within this study an experiment is uniquely defined by the combination of the following parameters:

| Parameter | Description                                           | Available Options                                           |
| --------- | ----------------------------------------------------- | ----------------------------------------------------------- |
| method    | The few-shot learning method to use                   | `baseline`, `baseline++`, `protonet`, `matchingnet`, `maml` |
| dataset   | The dataset to use                                    | `tabula_muris`, `swissprot`                                 |
| use_sot   | Whether the SOT feature transform should be used      | `True`, `False`                                             |
| n_way     | The number of classes in the few-shot task            |                                                             |
| n_shot    | The number of examples per class in the few-shot task |                                                             |

Each experiment is run as part of a set of experiments (An example of this can be seen in [test.sh](./test.sh)) which specifies a grid of experiment configurations to run. The experiment group is defined by the `group` parameter. We use [Hydra](https://hydra.cc) to configure all experiments - a single experiment can be run using the following command:

```bash
python run.py group={group} method={method} dataset={dataset} use_sot={use_sot} n_way={n_way} n_shot={n_shot}
```

To automatically run a hyperparameter search for any given experiment combination simply add the `--multirun` flag to the command above. This will run the experiment for all combinations of the parameters that are valid for the given experiment.

```bash
python run.py --multirun group={group} method={method} dataset={dataset} use_sot={use_sot} n_way={n_way} n_shot={n_shot}
```

*Note, that depending on the method used, the hyperparameters may vary, e.g. MAML has some method-specific hyperparameters.*

As a general rule, none of the options which are configured in `/conf` have to be changed (reasonable defaults) but in some cases changing them dynamically might be useful (e.g. disable logging behaviour for debugging or running the experiments on a different device). For a full list of available configurations, look at the [conf](./conf) directory.

## 📊 Experiment Tracking

We use [Weights and Biases](https://wandb.ai/) (WandB) for tracking experiments and results during training and evaluation. By default, we log the experiment setup (as specified by the Hydra configuration) and the training loss, validation accuracy and the few-shot accuracy on all splits for the best-performing model, as well as the best performing model itself as an artifact.

*Note, that W&B can be disabled using `wandb.mode=disabled` from the command line.*
