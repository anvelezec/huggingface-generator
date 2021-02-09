# coding=utf-8
# Copyright 2020 HuggingFace Datasets Authors.
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

# Lint as: python3
"""Language-Independent Named Entity Recognition"""

import logging

import datasets


_CITATION = "Velez A (2020)"


_DESCRIPTION = "This dataset aims to dispose information to train a named entity recognition model \
    with focus in financial sector."

_URL = "data/"
_TRAINING_FILE = "train.txt"
_DEV_FILE = "dev.txt"
_TEST_FILE = "test.txt"


class FinancieroConfig(datasets.BuilderConfig):
    """BuilderConfig for financiero"""

    def __init__(self, **kwargs):
        """BuilderConfig for Financiero.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(NerKConfig, self).__init__(**kwargs)


class Financiero(datasets.GeneratorBasedBuilder):
    """Finaniciero dataset."""

    BUILDER_CONFIGS = [
        NerKConfig(name="Financiero", version=datasets.Version("1.0.0"), description="Financiero"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=[
                                "O",
                                "B-REI",
                                "B-PII",
                                "B-PRO",
                                "B-PRC",
                                "B-CON",
                                "B-SUG",
                                "B-NUM",
                                "B-NB",
                                "B-LR",
                                "B-VA",
                                "B-TPO",
                                "B-JTO",
                                "I-REI",
                                "I-PII",
                                "I-PRO",
                                "I-PRC",
                                "I-CON",
                                "I-SUG",
                                "I-NUM",
                                "I-NB",
                                "I-LR",
                                "I-VA",
                                "I-TPO",
                                "I-JTO",
                                "E-REI",
                                "E-PII",
                                "E-PRO",
                                "E-PRC",
                                "E-CON",
                                "E-SUG",
                                "E-NUM",
                                "E-NB",
                                "E-LR",
                                "E-VA",
                                "E-TPO",
                                "E-JTO",
                                "S-REI",
                                "S-PII",
                                "S-PRO",
                                "S-PRC",
                                "S-CON",
                                "S-SUG",
                                "S-NUM",
                                "S-NB",
                                "S-LR",
                                "S-VA",
                                "S-TPO",
                                "S-JTO"
                                ]
                        )
                    ),
                }
            ),
            supervised_keys=None,
            homepage="XXX",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        urls_to_download = {
            "train": _URL + _TRAINING_FILE,
            "dev": _URL + _DEV_FILE,
            "test": _URL + _TEST_FILE,
        }
        downloaded_files = dl_manager.extract(urls_to_download)

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files["dev"]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": downloaded_files["test"]}),
        ]

    def _generate_examples(self, filepath):
        logging.info("Generating examples from = %s", filepath)
        with open(filepath, encoding="latin1") as f:
            guid = 0
            tokens = []
            ner_tags = []
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if tokens:
                        yield guid, {
                            "id": str(guid),
                            "tokens": tokens,
                            "ner_tags": ner_tags,
                        }
                        guid += 1
                        tokens = []
                        ner_tags = []
                else:
                    # Financiero tokens are space separated
                    try:
                        splits = line.split(" ")
                        tokens.append(splits[0])
                        ner_tags.append(splits[1].replace("\n", "").rstrip())
                    except:
                        print(splits)
            # last example
            yield guid, {
                "id": str(guid),
                "tokens": tokens,
                "ner_tags": ner_tags,
            }