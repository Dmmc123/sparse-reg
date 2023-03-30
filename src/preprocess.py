from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path
import argparse
import scipy
import spacy
import json
import re


class Preprocessor:

    def __init__(self) -> None:
        """Initialize preprocessing engines"""
        self.vectorizer = TfidfVectorizer(
            tokenizer=self._tokenize,
            token_pattern=None
        )
        self.lemma_model = spacy.load('en_core_web_sm')
        self.token_regex = r"[\w\d]+"

    def _tokenize(self, text: str) -> list[str]:
        """
        Transform a raw string into list of lemmatized tokens

        :param str text: Reddit post text
        :return: List of tokens
        :rtype: list[str]
        """
        tokens = re.findall(self.token_regex, text)
        return self._lemmatize(tokens)

    def _lemmatize(self, tokens: list[str]) -> list[str]:
        """
        Lemmatize the extracted tokens

        :param tokens: List of candidate tokens
        :return: Lemmas of tokens
        :rtype: list[str]
        """
        glued_tokens = " ".join(tokens)
        doc = self.lemma_model(glued_tokens)
        return [token.lemma_ for token in doc]

    def clean(self, file_name: str, out_dir: str) -> None:
        """
        Vectorize texts from dataset using TF-IDF and save it into the specified folder

        :param str file_name: Path to the raw dataset
        :param str out_dir: Folder to dump the processed dataset
        """
        texts, targets = self._read_data(file_name)
        vectors = self.vectorizer.fit_transform(texts)
        vocabulary = self.vectorizer.get_feature_names_out().tolist()
        self._save_dataset(
            vectors=vectors,
            vocabulary=vocabulary,
            targets=targets,
            orig_file_name=file_name,
            out_dir=out_dir
        )

    @staticmethod
    def _read_data(file_name: str) -> tuple[list[str], list[int]]:
        """
        Read dataset on json format. Samples need to have the following fields:
            * text
            * score

        :param str file_name: Path to the dataset
        :return: Texts and corresponding score values of posts
        :rtype: tuple[list[str], list[int]]
        """
        with open(file_name, "r") as f:
            samples = json.load(f)
        texts = [sample["text"] for sample in samples]
        targets = [sample["score"] for sample in samples]
        return texts, targets

    @staticmethod
    def _save_dataset(vectors: scipy.sparse.csr_matrix,
                      vocabulary: list[str],
                      targets: list[int],
                      orig_file_name: str,
                      out_dir: str) -> None:
        """
        Save processed dataset in a special folder with following structure:
            * features.npz - sparse matrix of TF-IDF features
            * metadata.json - json file with fields "targets" and "vocabulary"

        :param list[list[float]] vectors: TF-IDF for each sample in dataset
        :param list[str] vocabulary: List of words in vocabulary
        :param list[int] targets: Scores of posts
        :param str orig_file_name: json filename of the raw dataset
        :param str out_dir: Folder to save processed dataset
        """
        # create folder for storing dataset
        file_name = orig_file_name.split("/")[-1].split(".")[0]
        folder = f"{out_dir}/{file_name}"
        Path(folder).mkdir(parents=True, exist_ok=True)
        # dump vectors
        scipy.sparse.save_npz(f"{folder}/features.npz", vectors)
        # dump metadata
        with open(f"{folder}/metadata.json", "w") as f:
            json.dump(
                obj={
                    "targets": targets,
                    "vocabulary": vocabulary
                },
                fp=f
            )


def main():
    # define arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--input-dir", "-in", required=True,
        help="Folder with input json datasets"
    )
    argparser.add_argument(
        "--output-dir", "-out", required=True,
        help="Folder for storing preprocessed datasets"
    )
    args = argparser.parse_args()
    # execute preprocessing jobs
    preprocessor = Preprocessor()
    for file_name in Path(args.input_dir).rglob("*.json"):
        preprocessor.clean(
            file_name=str(file_name),
            out_dir=args.output_dir
        )


if __name__ == "__main__":
    main()
