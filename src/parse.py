from dotenv import load_dotenv
from requests import auth
import argparse
import requests
import logging
import json
import os


class RedditParser:

    def parse(self, sub_reddit: str, out_folder: str, n_samples: int = 1_000, n_batch: int = 50) -> None:
        """
        Extract and save post info from a sub reddit

        :param str sub_reddit: Name of the subreddit of interest
        :param str out_folder: Folder for saving the dataset
        :param int n_samples: How many posts to parse from a sub-reddit
        :param int n_batch: How many posts to request in a single batch
        """
        samples = self._get_samples(
            sub_reddit=sub_reddit,
            n_samples=n_samples,
            n_batch=n_batch
        )
        self._save_samples(
            samples=samples,
            sub_reddit=sub_reddit,
            out_folder=out_folder
        )

    def _get_samples(self, sub_reddit: str, n_samples: int, n_batch: int) -> list[dict]:
        """
        Use Reddit API to get post data in specified sample format

        :param str sub_reddit: Name of subreddit to parse
        :param int n_samples: Amount of samples to parse and return
        :param int n_batch: How many posts to request in a single batch
        :return: List of not preprocessed samples from post Reddit data
        :rtype: list[dict]
        """
        headers = self._init_connect()
        params = {"limit": n_batch}
        samples = []
        while len(samples) < n_samples:
            try:
                batch, params = self._get_batch_samples(
                    sub_reddit=sub_reddit,
                    headers=headers,
                    params=params
                )
            except IndexError:
                logging.log(level=logging.INFO, msg=f"Stopped at {len(samples)} samples")
                break
            samples.extend(batch)
        return samples[:n_samples]

    @staticmethod
    def _save_samples(samples: list[dict], sub_reddit: str, out_folder: str) -> None:
        """
        Save a list of samples in json format

        :param list[dict] samples:
        :param str sub_reddit: Name if the subreddit. Will act as a json filename
        :param str out_folder: Folder for storing Reddit datasets
        """
        if not samples:
            raise ValueError(f"Provided subreddit {sub_reddit} does not exist!")
        with open(f"{out_folder}/{sub_reddit}.json", "w") as f:
            json.dump(samples, f)

    @staticmethod
    def _init_connect() -> dict[str, str]:
        """
        Initialize connection with Reddit API and return headers for subsequent requests.
        Relies on following environmental variables:
            * REDDIT_APP_ID
            * REDDIT_APP_SECRET
            * REDDIT_APP_NAME
            * REDDIT_USERNAME
            * REDDIT_PASSWORD

        :return: Dictionary of headers
        :rtype: dict[str, str]
        """
        reddit_auth = auth.HTTPBasicAuth(
            os.environ["REDDIT_APP_ID"],
            os.environ["REDDIT_APP_SECRET"],
        )
        login_params = {
            "grant_type": "password",
            "username": os.environ["REDDIT_USERNAME"],
            "password": os.environ["REDDIT_PASSWORD"],
        }
        headers = {"User-Agent": f"{os.environ['REDDIT_APP_NAME']}/0.0.1"}
        # make initial request for token
        resp = requests.post(
            url="https://www.reddit.com/api/v1/access_token",
            auth=reddit_auth,
            data=login_params,
            headers=headers
        )
        token = resp.json()["access_token"]
        # add token to headers
        headers["Authorization"] = f"bearer {token}"
        return headers

    def _get_batch_samples(self, sub_reddit: str, headers: dict, params: dict) -> tuple[list[dict], dict]:
        """
        Extract a batch of samples and take the fields of interest

        :param str sub_reddit: Name of sub-reddit for url construction
        :param dict headers: Authorization headers
        :param dict params: Limits and search-after params
        :return:
            Batch of samples with extracted essential fields
            and parameters for next batch request
        :rtype: tuple[list[dict], dict]
        """
        url = f"https://oauth.reddit.com/r/{sub_reddit}/hot"
        resp = requests.get(url=url, headers=headers, params=params).json()
        params = self._construct_next_params(response=resp, params=params)
        samples = self._extract_samples(response=resp)
        return samples, params

    @staticmethod
    def _construct_next_params(response: dict, params: dict) -> dict:
        """
        Update current parameters for querying next batch of samples

        :param dict response: Raw response from Reddit API
        :param params: Parameters used for request
        :return: Updated version of parameters for subsequent request
        :rtype: dict
        """
        last_post = response["data"]["children"][-1]
        params["after"] = f"{last_post['kind']}_{last_post['data']['id']}"
        return params

    def _extract_samples(self, response: dict) -> list[dict]:
        """
        Process raw response and extract post data

        :param dict response: Raw response with posts from Reddit API
        :return: List of posts with extracted fields
        :rtype: list[dict]
        """
        return [self._extract_sample(child) for child in response["data"]["children"]]

    @staticmethod
    def _extract_sample(child: dict) -> dict:
        """
        Extract necessary fields from json post

        :param dict child: Raw information about post from response body
        :return: Single processed post
        :rtype: dict
        """
        return {
            "text": f"{child['data']['title']}. {child['data']['selftext']}",
            "score": child['data']["score"]
        }


def main():
    # define arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--sub-reddits", "-sr", nargs="+", required=True,
        help="List of subreddits (e.g. python chess starwars)"
    )
    argparser.add_argument(
        "--output-dir", "-dir", required=True,
        help="Folder for storing downloaded datasets"
    )
    argparser.add_argument(
        "-env", required=False, default=".env",
        help="Path to environmental file .env",
    )
    args = argparser.parse_args()
    # execute parsing jobs
    load_dotenv(args.env)
    parser = RedditParser()
    for sub_reddit in args.sub_reddits:
        parser.parse(sub_reddit, args.output_dir)


if __name__ == "__main__":
    main()
