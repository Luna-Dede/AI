from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType

def calculate_similarity(list1, list2):
    """
    Given two lists of relevance scores, calculate sim score. 
    Returns [float]: Similarity score
    """
    
    if len(list1) != len(list2):
        logging.error("List lengths are not equal, similarity calculation failed.")
        raise ValueError("Lists must be of equal length")

    similarity_score = 0
    for i in range(len(list1)):
        similarity_score += abs(list1[i] - list2[i]) * 25
    
    return similarity_score / len(list1)



calculate_similarity_udf = udf(calculate_similarity, FloatType())
