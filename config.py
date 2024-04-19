from pyspark.sql import SparkSession
import logging


# Paths to data files
GENOME_SCORES_FILE = 'data/genome-scores.csv'
RATINGS_FILE = 'data/ratings.csv'
OUTPUT_DIRECTORY = 'data/output/'

# Spark configuration settings
SPARK_APP_NAME = "Movie Similarity"
SPARK_EXECUTOR_MEMORY = "7g"
SPARK_DRIVER_MEMORY = "4g"
SPARK_EXECUTOR_INSTANCES = "16"
SPARK_EXECUTOR_CORES = "4"
SPARK_DEFAULT_PARALLELISM = "64"
SPARK_SHUFFLE_PARTITIONS = "64"


def config_logging():
    """
    Initialize 
    """
    logging.basicConfig(filename='Spark_similarity_computation.log', 
                    level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

    # initialize logging
    logging.info('Logging system initialized')




def config_SparkSession(appName=SPARK_APP_NAME):
    
    """
    Configures and returnsSpark Session
    params: AppName [str]: Name of spark session (default Movie Rec. Engine)
    Return: Spark session object 
    """


    spark = SparkSession.builder \
    .appName("Movie Similarity") \
    .config("spark.executor.memory", SPARK_EXECUTOR_MEMORY) \
    .config("spark.driver.memory", SPARK_DRIVER_MEMORY) \
    .config("spark.executor.instances", SPARK_EXECUTOR_INSTANCES) \
    .config("spark.executor.cores", SPARK_EXECUTOR_CORES) \
    .config("spark.default.parallelism", SPARK_DEFAULT_PARALLELISM) \
    .config("spark.sql.shuffle.partitions", SPARK_SHUFFLE_PARTITIONS) \
    .getOrCreate()

    return spark
