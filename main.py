import os
import pyspark
import pandas
import numpy
import logging
from scipy.special import softmax
from tqdm import tqdm
from config import config_logging, config_SparkSession
from config import GENOME_SCORES_FILE, RATINGS_FILE, OUTPUT_DIRECTORY
from model import q_learning
from data_processing import process_ratings, process_gen_scores




def main():

    # initialize logging and Spark Session
    config_logging()
    spark = config_SparkSession()

    # process genome scores & ratings file (RETURNS SPARK DF)
    ratings_df = process_ratings(spark, RATINGS_FILE)
    sim_scores = process_gen_scores(spark, GENOME_SCORES_FILE)

    # cache ratings and sim df's
    logging.info("Caching dataframes")
    ratings_df.cache()
    sim_scores.cache()
    logging.info("Caching complete.")


    # Log the start of model processing
    logging.info("Starting the Q-learning model computation.")
    
    # run model
    num_episodes = 3000 
    Q, optimal_policy, avg_rewards = q_learning(
        spark, num_episodes, ratings_df, sim_scores)

    logging.info("Q-learning model computation completed.")

    # write out
    pd.DataFrame(Q).to_csv(f'{OUTPUT_DIRECTORY}/newqtable2.csv')
    pd.DataFrame(opt).to_csv(f'{OUTPUT_DIRECTORY}/new_policy2.csv')
    pd.DataFrame(np.array(rewards)).to_csv(f'{OUTPUT_DIRECTORY}/rewards2.csv')


    logging.info("Output files have been written to the disk.")






if __name__ == "__main__":
    main()


