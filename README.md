<picture>
  <source media="(prefers-color-scheme: dark)" srcset=".github/assets/banner-dark.png">
  <img alt="The University of Chicago, MS in Applied Data Science. ADSP 32014 course project." src=".github/assets/banner-light.png">
</picture>

# Book Recommendation System Using Bayesian Machine Learning

> **Example project from Winter 2025.** This is a copy of [BadgerDaniel/MasterBayesianBookRec](https://github.com/BadgerDaniel/MasterBayesianBookRec), shown with the authors' work and commit history intact. The authors' own write-up is [Readme_Bayesian.ipynb](Readme_Bayesian.ipynb). Course staff wrote this README and made the edits listed under [Changes by course staff](#changes-by-course-staff), including the section on [what to learn from this project](#what-to-learn-from-this-project).

Can a recommender say not only which books a reader will like, but how sure it is, and use that uncertainty to decide what to show next? The team built a Bayesian collaborative filtering model on 380,000 Book-Crossing ratings and ranked books by expected rating plus a bonus for books the model is unsure about.

## Summary

The pipeline runs end to end. A data-preparation notebook cleans the Kaggle Book-Crossing data, reconciling the many ISBNs that point to the same title and filling missing publication years through the Google Books API. A PyMC model represents each reader and each book with a bias and a short latent vector, with hierarchical priors, and models ratings with a Poisson likelihood. Four prior hyperparameters were tuned by Bayesian optimization, and the model was fit with NUTS on a GPU, and in a second variant with ADVI. For each reader, posterior samples give an expected rating and a variance for every book, and a ranking rule adds an exploration bonus to uncertain books when the gap to the best book is large enough. A Streamlit app shows a reader's top-rated books next to the model's recommendations.

The team ran out of compute before the model could be fit on enough data, so the reported accuracy is low. The section on [what to learn from this project](#what-to-learn-from-this-project) explains why, and what the project does demonstrate.

## Team and contributions

- **Daniel Sa** ([@BadgerDaniel](https://github.com/BadgerDaniel))
- **Samuel Fisher** ([@samuelfisher927](https://github.com/samuelfisher927))
- **Hyunji (Amy) Kim** ([@hyunji0618](https://github.com/hyunji0618))
- **Jazil Kalim** ([@jazilkalim](https://github.com/jazilkalim))

The project ran from January 29 to March 13, 2025, with each member on their own branch. The [commit history](https://github.com/uchicago-dsi-bayesianml/wi25-bayesian-book-rec/commits/main) and the section labels in the write-up record who did what.

## Getting started

The project has three environments, one per stage. Each stage has its own requirements file.

1. **Data preparation** (`DataCleaning_Prep_EDA/`, Python 3.10, `dataprep_requirements.txt`). Download the dataset described under [Data](#data) and unzip `Books.csv`, `Ratings.csv` and `Users.csv` next to `DataPrep.ipynb`. Running the notebook writes `books_updated.csv`, `ratings_updated.csv` and `book_ratings_cleaned.csv`. The two mapping files in the folder, `isbn_mapping.csv` and `isbn_year_mapping.csv`, are the saved results of the deduplication and the Google Books lookups, so you don't need an API key to reproduce the cleaning.

2. **Modeling** (`Modeling/`, `modeling_requirements.txt`). `ModelFinetuning_and_Training_MBBR.ipynb` was written for Google Colab with a GPU: it installs PyMC with the NumPyro sampler, tunes the hyperparameters, fits the final model and saves the trace as a NetCDF file. The tuned values are in `parameters/tuned_hyperparameters.txt`. The trained traces are too large for GitHub and were shared by the authors from [Google Drive](https://drive.google.com/drive/folders/1DbUQNgEGAtgdPEnzKIz93IVDIqvNarjS?usp=sharing).

3. **The app** (`UI/`, `requirements.txt`). From the repository root, with `books_updated.csv` and `book_ratings_cleaned.csv` copied there:

   ```sh
   pip install -r UI/requirements.txt
   streamlit run UI/BayesianBookRecUI_v1.py
   ```

   The app reads `book_recommendations2.nc`, the saved top-3 recommendations per reader, which is small enough to stay in the repository. The authors' deployment is at [masterbayesianbookrec.streamlit.app](https://masterbayesianbookrec.streamlit.app/); Streamlit puts idle apps to sleep, so it may take a minute to wake.

## Data

Data files are never committed to this repository. Git ignores the raw download and the cleaned tables.

- **Source:** the [Book Recommendation Dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset) on Kaggle, a repackaging of the Book-Crossing dataset collected by Cai-Nicolas Ziegler in 2004: about 1.1 million ratings of 270,000 books by 280,000 readers, of which 384,000 are explicit ratings on a 1 to 10 scale. A rating of 0 means an implicit interaction and is dropped before modeling.
- **Terms:** public domain (CC0) on Kaggle. Book cover images in the app are fetched live from Amazon and Bing and aren't stored.
- **How to get it:** download the three CSV files from Kaggle and place them next to `DataPrep.ipynb`.
- **Where to put it:** the cleaned outputs go in the repository root for the app, as described above.

## Project structure

| Path | Contents |
|---|---|
| `Readme_Bayesian.ipynb` | The authors' write-up: introduction, model, ranking strategy, results, takeaways. |
| `DataCleaning_Prep_EDA/DataPrep.ipynb` | Exploration and cleaning: ISBN reconciliation, year imputation, the three output tables. |
| `DataCleaning_Prep_EDA/isbn_mapping.csv`, `isbn_year_mapping.csv` | Saved results of the deduplication and the Google Books lookups. |
| `Modeling/ModelFinetuning_and_Training_MBBR.ipynb` | The PyMC model, Bayesian optimization of its hyperparameters, and the final NUTS fit. |
| `Modeling/parameters/` | The tuned hyperparameters. |
| `Future Improvements/Bayesian_book_model-AlternativeMetrics.ipynb` | The ADVI variant, the ranking rule, and ranking metrics (MRR, NDCG, coverage, novelty). |
| `Future Improvements/archive/` | The project proposal and an early notebook. |
| `UI/BayesianBookRecUI_v1.py` | The Streamlit app. |
| `book_recommendations2.nc` | Saved top-3 recommendations per reader, read by the app. |
| `.devcontainer/` | A Codespaces configuration that launches the app. |

## Method

**Model.** For reader $u$ and book $i$, the rating is Poisson with rate

$$\lambda_{ui} = \exp\big(\mu + b_u + b_i + \alpha_u^\top \beta_i\big)$$

where $\mu$ has a Gamma prior, the biases $b_u$ and $b_i$ have Normal priors whose scale shrinks with the number of ratings a reader or book has, and the latent vectors $\alpha_u$ and $\beta_i$ have Normal priors with Half-Cauchy hyperpriors on their scales. The latent dimension is 3 in the final NUTS model and 5 in the ADVI variant.

**Inference.** Four hyperparameters, the two Half-Cauchy scales and the Gamma prior's shape and rate, were tuned with Gaussian-process Bayesian optimization. The final model was sampled with NUTS through NumPyro on a GPU, two chains of 500 draws after 500 tuning steps. The variant in `Future Improvements/` uses ADVI for 50,000 iterations and draws 2,000 samples from the fitted approximation.

**Ranking.** For a reader, every book's rating is predicted under each posterior sample, giving a mean and a variance per book. The regret of a book is the gap between the best mean and its own. Books whose regret exceeds a threshold receive an exploration bonus proportional to their variance, and the top five by mean plus bonus are recommended.

### What is Bayesian here

The model is a hierarchical Bayesian matrix factorization. Every reader bias, book bias and latent vector has a prior, and the scales of those priors are themselves unknowns with hyperpriors, so the data decide how much readers and books are allowed to differ. The count-dependent prior scales are a hand-built form of shrinkage: a reader with two ratings stays close to the average, a reader with two hundred can move away from it. Inference produces a posterior over all of these at once, by NUTS or by a variational approximation, and the ranking rule uses that posterior rather than a point estimate. The spread of predicted ratings across posterior samples is the model's uncertainty about a book for that reader, and the exploration bonus turns it into a decision: recommend something uncertain when the expected cost of doing so is small. That is the exploration-exploitation trade-off from Bayesian decision theory, applied to a recommender.

## Results

The authors report, for the final model, a training RMSE of 1.44 and a test RMSE of 2.58, with precision and recall around 60% on the training split and around 20% on the test split at a rating threshold of 7. The ADVI variant reaches a test RMSE of 4.4 and a precision at 5 of 0.001 on held-out readers. The authors' takeaways in the write-up are about priors, the order of tuning, and the cost of sampling, and they note the model overfits and was trained on a subset for lack of compute.

## What to learn from this project

Course staff added this section. The project is worth studying for its scope and for one mistake that is easy to make.

- **The subsample broke the model.** For compute reasons the model was fit on 1,000 ratings drawn at random from 384,000. Those 1,000 ratings involve 838 readers and 977 books, so almost every reader and book appears once. A matrix factorization learns from readers and books that recur, and here almost nothing recurs, which is why the test metrics are near chance. The fix would have cost nothing: keep the few hundred most active readers and most rated books, which gives a small but dense matrix the same hardware can handle.
- **No baseline.** The write-up names SVD as the point of comparison, but no non-Bayesian baseline was run, so there is no way to tell what the uncertainty bought.
- **The write-up and the code disagree in places.** Section 3 of the write-up describes a model with a Normal variance and Normal priors on the global mean, and section 4.6 describes gradient-descent parameter updates. The code implements the Poisson model described under [Method](#method) and fits it by NUTS and ADVI. Read the code for what was actually run.
- **"Multi-step lookahead" is a one-step rule.** The ranking adds an uncertainty bonus to the current expected rating. It doesn't simulate how a recommendation would change the posterior, which is what lookahead means. The idea behind it, exploring when the regret of doing so is small, is sound.
- **The precision and recall figures are the same number.** The evaluation computes micro-averaged precision and recall on binary labels, which both equal accuracy. That is why the pairs in the write-up match.

What the project does demonstrate well: a careful data-cleaning stage with saved, reproducible mappings; a correctly specified hierarchical model in PyMC with both sampling and variational inference; hyperparameter tuning treated as its own optimization problem; posterior samples used for a decision rule rather than a point prediction; and a working app on top of it.

## About this project

This project was built in Bayesian Machine Learning with Generative AI Applications (ADSP 32014), part of the Master of Science in Applied Data Science at the University of Chicago, in Winter 2025. It is shown here as an example for later cohorts.

## Changes by course staff

Course staff forked the authors' repository and made these edits. The authors' commits and notebooks are unchanged.

- Added this README and the program banner. The authors' own write-up is `Readme_Bayesian.ipynb`.
- Removed data files from the tree: the Kaggle download, and the cleaned tables that had been committed three times over, about 300 MB. They remain in the history and are recreated by `DataPrep.ipynb`. The two small ISBN mapping files and the saved recommendations stay.
- Replaced two Word files holding the tuned hyperparameters with `Modeling/parameters/tuned_hyperparameters.txt`, removed a scratch notebook with machine-specific paths, and re-encoded `UI/requirements.txt` from UTF-16 to UTF-8 so GitHub displays it.

## License

The authors have not added a license, so their code is shown here under GitHub's terms for public repositories. Ask them before reusing it.
