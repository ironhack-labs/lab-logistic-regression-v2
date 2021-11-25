![logo_ironhack_blue 7](https://user-images.githubusercontent.com/23629340/40541063-a07a0a8a-601a-11e8-91b5-2f13e4e6b441.png)

# Lab | Logistic Regression (v2)

Use the Sakila database. In this lab, you will have to generate a logistic regression model to predict if the rating of a movie will be any of ['G','PG','PG-13'] or not (['NC-17','R']), based on the movie description.
To do this follow the steps below:

1. Determine the SQL query to obtain for each movie, the description and the rating.
2. Create a new Jupyter notebook, establish a connection with the **sakila** database. 
3. Create a Python function to retrieve the data from the database given the engine from the previous query.
4. Create a Python function named `binary_rating` which will replace the `rating` values by 'Yes' or 'No' whether the movie rating is in ['G','PG','PG-13'] or not (['NC-17','R']).
5. Create a Python function name `get_df_corpus` that given the dataframe,will return a list in which each element will be a movie description. Store the function returned list as `corpus` for later.
6. Do the data splitting (ie. set the X and the y).
7. Do the train-test split.
8. Now what you need to create a model will be to dummify the words appearing in each description (ignoring stop-words). This can be done with the following chunk of code
  ```python
  from sklearn.feature_extraction.text import CountVectorizer
  # Here we set the option stop_words = 'english' to take into account the 'stop_words' in English. 
  # Other languages have different stop_words.
  # We also set the option analyzer='word' to analyze words.
  # See the additional resources section for more information
  vectorizer = CountVectorizer(stop_words = 'english', analyzer='word')
  vectorizer.fit(corpus)

  # Transforming descriptions to arrays of words counts
  X_train_counts = vectorizer.transform(X_train)
  X_test_counts  = vectorizer.transform(X_test)

  # Working with counts can be misleading for a model. It's better to work with weighted word frequencies 
  # The idea is: count how many times appear each word in each description, and then compensate by the inverse
  # of the number of times that this word appears in all the descriptions.
  # See the additional resources section for more information
  from sklearn.feature_extraction.text import TfidfTransformer

  tf_transformer = TfidfTransformer()
  tf_transformer.fit(X_train_counts)
  X_train_tfidf = tf_transformer.transform(X_train_counts)
  X_test_tfidf  = tf_transformer.transform(X_test_counts)
  ```
9. Train a logistic regression model using `X_train_tfidf` and `y_train` as input.
10. Get the rating predictions for the `X_train_tfidf` and `X_test_tfidf`.
11. Use pickle to save: the vectorizer, the tf_transformer in a folder named `transformers` and the model a folder named `models`.