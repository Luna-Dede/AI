from pyspark.sql import functions as F
from pyspark.sql.functions import col, broadcast
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.special import softmax
import logging
import time






def q_learning(spark, num_episodes, ratings, similarity_scores):
    # Initialize variables
    movies = [row['newMovieId'] for row in ratings.select('newMovieId').distinct().collect()]
    movie_index = {movie: idx for idx, movie in enumerate(movies)}
    
    # Create reward dictionary
    reward_dict = {0.5: -25, 1: -20, 1.5: -15, 2: -10, 2.5: -5, 3: 0, 3.5: 5, 4: 10, 4.5: 15, 5: 25}

    # Initialize Q-table and other parameters
    Q = np.zeros((len(movies), len(movies)))
    gamma = 0.05
    epsilon = 0.9
    optimal_policy = np.random.choice(movies, size=len(movies))

    num_updates = np.zeros((len(movies), len(movies)))
    avg_rewards = []

    # Begin episodes
    for episode in range(num_episodes):
        # Sample initial observation
        observation = ratings.sample(False, 0.1).limit(1).collect()[0]
        mov = movie_index[observation['newMovieId']]
        userId = observation['userId']

        terminated = False
        cum_reward = 0
        num_steps = 0

        while not terminated:
            # Fetch possible actions and their precomputed similarity scores
            possible_actions = similarity_scores.filter(similarity_scores.movieId1 == observation['newMovieId'])
            if possible_actions.count() == 0:
                break

            # Decision making: either explore or exploit
            if np.random.rand() < epsilon:
                action = possible_actions.orderBy(F.rand()).limit(1).collect()[0]
            else:
                action = possible_actions.orderBy(F.desc('similarity')).limit(1).collect()[0]

            action_mov = movie_index[action['movieId2']]
            sim_score = action['similarity']

            # Fetch new state and reward
            new_state = ratings.filter((ratings.userId == userId) & (ratings.movieId == action['movieId2'])).collect()
            if len(new_state) == 0:
                break
            new_state = new_state[0]

            reward = reward_dict[new_state['rating']]
            cum_reward += reward

            # Q-table update
            eta = 1 / (1 + num_updates[mov, action_mov])
            max_next_q = np.max(Q[action_mov, :])  # Assuming Q is still maintained locally
            Q[mov, action_mov] = (1 - eta) * Q[mov, action_mov] + eta * (reward + gamma * max_next_q)
            num_updates[mov, action_mov] += 1

            # Update policy and state for next iteration
            optimal_policy[mov] = np.argmax(Q[mov, :])
            observation = new_state
            mov = action_mov
            num_steps += 1

            if reward < 15 or num_steps > 20:
                avg_reward = cum_reward / num_steps if num_steps != 0 else 0
                avg_rewards.append(avg_reward)
                terminated = True

        # Decay epsilon
        epsilon *= 0.9997

    return Q, optimal_policy, avg_rewards


"""

def q_learning(spark, num_episodes, ratings, similarity_scores):
    movies_df = ratings.select('newMovieId').distinct()
    movies = [row['newMovieId'] for row in movies_df.collect()]
    movie_index = {movie: idx for idx, movie in enumerate(movies)}
    broadcast_movie_index = spark.sparkContext.broadcast(movie_index)
    
    mean_ratings = ratings.groupBy("newMovieId").avg("rating").rdd.collectAsMap()
    Q = np.zeros((len(movies), len(movies)))
    for movie, idx in movie_index.items():
        mean_rating = mean_ratings.get(movie, 0.5)
        Q[:, idx] = (mean_rating - 0.5) * 10 - 25

    reward_dict = {0.5: -25, 1: -20, 1.5: -15, 2: -10, 2.5: -5, 3: 0, 3.5: 5, 4: 10, 4.5: 15, 5: 25}
    broadcast_rewards = spark.sparkContext.broadcast(reward_dict)

    gamma = 0.05
    epsilon = 0.9
    num_updates = np.zeros((len(movies), len(movies)))
    optimal_policy = np.random.choice(movies, size=len(movies))
    avg_rewards = []

    for episode in range(num_episodes):
        logging.info(f"Episode: {episode + 1}")
        start_time = time.time()
        observation = ratings.sample(False, 0.1).limit(1).collect()[0]
        mov_idx = broadcast_movie_index.value.get(observation['newMovieId'])
        if mov_idx is None:
            continue
        
        userId = observation['userId']
        terminated = False
        cum_reward = 0
        num_steps = 0

        logging.info(f"Getting Actions")
        actions_df = similarity_scores.filter(F.col("movieId1") == observation['newMovieId'])
        actions_df.persist()

        while not terminated:
            action = actions_df.orderBy(F.rand() if np.random.rand() < epsilon else F.desc('similarity')).limit(1).collect()[0]
            action_idx = broadcast_movie_index.value.get(action['movieId2'])
            if action_idx is None:
                continue
            
            new_state = ratings.filter((F.col("userId") == userId) & (F.col("newMovieId") == action['movieId2'])).limit(1).collect()
            if not new_state:
                break

            reward = broadcast_rewards.value[new_state[0]['rating']]
            cum_reward += reward
            eta = 1 / (1 + num_updates[mov_idx, action_idx])
            max_next_q = np.max(Q[action_idx, :])
            Q[mov_idx, action_idx] = (1 - eta) * Q[mov_idx, action_idx] + eta * (reward + gamma * max_next_q)
            num_updates[mov_idx, action_idx] += 1

            optimal_policy[mov_idx] = np.argmax(Q[mov_idx, :])
            observation = new_state[0]
            num_steps += 1

            if reward < 15 or num_steps > 20:
                avg_rewards.append(cum_reward / num_steps)
                terminated = True

        epsilon *= 0.9997
        episode_time = time.time() - start_time
        print(f"Episode {episode + 1}/{num_episodes} completed in {episode_time:.2f} seconds")

        actions_df.unpersist()

    return Q, optimal_policy, avg_rewards"""