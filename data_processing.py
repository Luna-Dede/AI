from pyspark.sql.functions import col, collect_set, size, monotonically_increasing_id, udf, sort_array, collect_list
from pyspark.sql.types import FloatType
from pyspark.sql import Row
from pyspark.sql import functions as F
from config import config_logging
from utils import *
import logging



def process_ratings(spark, ratings_path, min_move=1000, min_use=50):
    """
    Set up ratings using Spark to handle large datasets efficiently.
    
    Parameters:
    ratings_df (DataFrame): Spark DataFrame containing ratings with 'movieId' and 'userId'.
    min_move (int): Minimum number of ratings per movie.
    min_use (int): Minimum number of ratings per user.

    Returns:
    DataFrame: Filtered and transformed ratings DataFrame.
    """

    # Log the start of the function
    logging.info("Setting up ratings...")

    # read ratings file
    ratings_df = spark.read.csv(ratings_path, header=True, inferSchema=True)

    # Group by movieId and filter based on the minimum number of ratings
    by_movie = ratings_df.groupBy('movieId').count()
    by_movie = by_movie.filter(by_movie['count'] >= min_move).select('movieId')

    # Group by userId and filter based on the minimum number of ratings
    by_user = ratings_df.groupBy('userId').count()
    by_user = by_user.filter(by_user['count'] >= min_use).select('userId')

    # Filter ratings based on the filtered lists of movies and users
    ratings_filtered = ratings_df.join(by_movie, 'movieId').join(by_user, 'userId')

    # Create a new movie index
    movie_dict = ratings_filtered.select('movieId').distinct()\
        .withColumn('newMovieId', F.monotonically_increasing_id())

    # Join back to ratings to replace movieId with newMovieId
    ratings_final = ratings_filtered.join(movie_dict, 'movieId')\
        .select(ratings_filtered['userId'], ratings_filtered['rating'], movie_dict['newMovieId'])

    # Log completion of setup
    logging.info("Ratings setup complete.")

    return ratings_final





def process_gen_scores(spark, gen_path):
    """
    Process genome scores by aggregating relevance scores, filtering valid movies,
    and preparing data for similarity calculations.

    Parameters:
    spark (SparkSession): The Spark session object.
    file_path (str): The file path to the genome scores CSV file.

    Returns:
    DataFrame: Similarity score Spark dataframe with cols:
                - movieId1
                - movieId2
                - Similarity (Sim score between movieId1 & movieId2)
    """
    # Read genome scores data
    gen_scores = spark.read.csv(gen_path, header=True, inferSchema=True)

    # Aggregate relevance scores by movieId
    scored_movies = gen_scores.groupBy("movieId").agg(
        F.collect_list("relevance").alias("relevance_scores")
    )

    # Count distinct tags to ensure we filter movies with complete tag sets
    expected_tags_count = gen_scores.select("tagId").distinct().count()

    # Filter movies to ensure only those with a complete set of tags are considered
    valid_movies = scored_movies.filter(F.size("relevance_scores") == expected_tags_count)

    # Assign unique IDs to each movie for pairing in similarity calculations
    movies_df = valid_movies.withColumn("id", F.monotonically_increasing_id())

    # Prepare DataFrame for cross-join by assigning unique IDs
    pairs_df = movies_df.alias("movies1").join(movies_df.alias("movies2"), F.col("movies1.id") < F.col("movies2.id"))

    # Calculate similarity using a user-defined function (UDF) or any appropriate method
    result_df = pairs_df.withColumn(
        "similarity",
        calculate_similarity_udf(F.col("movies1.relevance_scores"), F.col("movies2.relevance_scores"))
    )

    # Select the final structure of the DataFrame to output
    sim_scores = result_df.select(
        F.col("movies1.movieId").alias("movieId1"),
        F.col("movies1.id").alias("id1"),
        F.col("movies2.movieId").alias("movieId2"),
        F.col("movies2.id").alias("id2"),
        "similarity"
    )

    return sim_scores