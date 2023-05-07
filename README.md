# Reddit: Analysis of Langauge Patterns through Regularization

## Introduction

Welcome to my project! The primary goal of this project is to analyze Reddit posts and their corresponding upvotes using various regression models. My aim is to identify the relationship between the textual features of posts, such as their term frequency-inverse document frequency (TF-IDF), and their popularity as indicated by the number of upvotes.

I have structured the project to facilitate the entire process, from data acquisition to model evaluation. I provide a streamlined pipeline that enables users to easily download data from specified subreddits, preprocess the data to extract relevant features, train a variety of regression models, and evaluate their performance using several metrics.

In the subsequent sections, I will guide you through the setup and execution of the project, including a detailed explanation of the project structure, how to run the code, and the metrics used for evaluating model performance. Additionally, I provide a list of references to further explore the underlying concepts and algorithms employed in this project.

## Project Structure

```
├── datasets             <- Folder for storing datasets
|   ├── raw              <- Unprocessed data from subreddits: posts and upvotes
|   └── clean            <- TF-IDF of posts packed with upvote numbers
|
├── evaluation           <- Folder for metric tables and other visualization artifacts
├── refs                 <- Paper references for regularization algorithms
├── src                  <- Source code of the project (all scripts)
├── weights              <- Folder with serialized models
|
├── .gitignore           <- List of ignored files and directories
└── requirements.txt     <- List of dependencies of the project
```

## How to run?

1. Install the project dependencies:

```pip install -r requirements.txt```

2. The project uses [Reddit API](https://www.reddit.com/dev/api/) and relies on having `.env` file with the following variables:

|       Variable      | Description |
| ------------------- | ----------- |
| `REDDIT_APP_ID`     | Application ID on reddit dev portal               |
| `REDDIT_APP_SECRET` | Secret token for interaction with dev application |
| `REDDIT_APP_NAME`   | Alias of the reddit dev application               | 
| `REDDIT_USERNAME`   | Your username                                     |
| `REDDIT_PASSWORD`   | Your password                                     |

3. Download datasets from subreddits. You can specify any subreddits of your like as long as they exist:

```
python src/parse.py --sub-reddits [SUB_REDDIT_NAMES] 
                    --output-dir <folder_for_data> 
                    --env <path_to_env>
```

4. Once you have downloaded posts from your favourite sub-reddits, preprocess them to extract TF-IDF features:

```
python src/preprocess.py --input-dir <folder_with_raw_datasets> 
                         --output-dir <folder_to_output>
```

5. Once the data is preprocessed, you can train a bunch of simple regression models with one line:

```
python src/train.py --datasets-dir <processed_datasets_dir> 
                    --weights-dir <folder_to_save_weights>
```

6. Once the models were trained, you can run inference on them, compute the table of metrics, and draw the the word clouds to visualize important words to get more upvotes:

```
python src/evaluate.py --datasets-dir <processed_datasets_dir> 
                       --weights-dir <weights_folder> 
                       --output-dir <folder_to_save_artifacts>
```

## Visualization

Examples of words that positively correlate with number of upvotes in [/r/MachineLearning](https://www.reddit.com/r/MachineLearning/):

![Positively correlated words](evaluation/clouds/MachineLearning/mean_squared_error_pos.png)

Examples of words that negatively correlate with number of upvotes in [/r/chess](https://www.reddit.com/r/chess/):

![Negatively correlated words](evaluation/clouds/chess/mean_squared_error_neg.png)

Examples of feature importance plot for [/r/chess](https://www.reddit.com/r/chess/):

![Word importance](evaluation/rankings/chess/mean_squared_error_ranking.png)

## Metrics

<table>
  <thead>
    <tr>
      <th colspan="2">Subreddit</th>
      <th>Method</th>
      <th>MSE</th>
      <th>L1</th>
      <th>L2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan="2"></td>
      <td>MSE</td>
      <td>10.7</td>
      <td>318570.8</td>
      <td>6494.9</td>
    </tr>
    <tr>
      <td colspan="2">MachineLearning</td>
      <td>Lasso</td>
      <td>687.5</td>
      <td>278854.1</td>
      <td>6204.1</td>
    </tr>
    <tr>
      <td colspan="2"></td>
      <td>Ridge</td>
      <td>2261.8</td>
      <td>135578.7</td>
      <td>3752.9</td>
    </tr>
    <tr>
      <td colspan="2"></td>
      <td>MSE</td>
      <td>47.7</td>
      <td>455806.7</td>
      <td>8408.4</td>
    </tr>
    <tr>
      <td colspan="2">cscareerquestions</td>
      <td>Lasso</td>
      <td>95.9</td>
      <td>348375.5</td>
      <td>7510.0</td>
    </tr>
    <tr>
      <td colspan="2"></td>
      <td>Ridge</td>
      <td>4007.1</td>
      <td>152264.1</td>
      <td>3830.3</td>
    </tr>
    <tr>
      <td colspan="2"></td>
      <td>MSE</td>
      <td>13.4</td>
      <td>168560.1</td>
      <td>4583.4</td>
    </tr>
    <tr>
      <td colspan="2">compsci</td>
      <td>Lasso</td>
      <td>396.9</td>
      <td>138771.6</td>
      <td>4263.0</td>
    </tr>
    <tr>
      <td colspan="2"></td>
      <td>Ridge</td>
      <td>2819.2</td>
      <td>80455.7</td>
      <td>2736.6</td>
    </tr>
    <tr>
      <td colspan="2"></td>
      <td>MSE</td>
      <td>3373.2</td>
      <td>1200434.6</td>
      <td>25490.1</td>
    </tr>
    <tr>
      <td colspan="2">chess</td>
      <td>Lasso</td>
      <td>3869.9</td>
      <td>1056797.5</td>
      <td>24319.7</td>
    </tr>
    <tr>
      <td colspan="2"></td>
      <td>Ridge</td>
      <td>39884.5</td>
      <td>248024.9</td>
      <td>8363.5</td>
    </tr>
    <tr>
      <td colspan="2"></td>
      <td>MSE</td>
      <td>192.5</td>
      <td>229601.1</td>
      <td>3882.6</td>
    </tr>
    <tr>
      <td colspan="2">python</td>
      <td>Lasso</td>
      <td>208.8</td>
      <td>175321.1</td>
      <td>3509.7</td>
    </tr>
    <tr>
      <td colspan="2"></td>
      <td>Ridge</td>
      <td>1253.9</td>
      <td>77761.7</td>
      <td>1793.7</td>
    </tr>

  </tbody>
  <caption>Loss values for various regularization techniques and subreddits</caption>
</table>


## References

* [Zhang, C. H., & Zhang, T. (2012). A general theory of concave regularization for high-dimensional sparse estimation problems.](https://projecteuclid.org/journals/statistical-science/volume-27/issue-4/A-General-Theory-of-Concave-Regularization-for-High-Dimensional-Sparse/10.1214/12-STS399.pdf)
* [James Briggs (2020). How to Use the Reddit API in Python](https://towardsdatascience.com/how-to-use-the-reddit-api-in-python-5e05ddfd1e5c)
