import torch
import keyboard
import random
import numpy as np
from dequeue import Deque  
from game import SnakeGame, Direction, Point
from model import neural_network, neural_trainer
from graph import plot

# CONSTANTS FOR THE AGENT
MAX_MEMORY = 100_000  # MAX NUMBER OF EXPERIENCES THE AGENT CAN REMEMBER
BATCH_SIZE = 1000  # NUMBER OF EXPERIENCES SAMPLED FROM MEMORY DURING TRAINING
LR = 0.001  # LEARNING RATE OF THE NEURAL NETWORK

class Agent:
    def __init__(self):
        self.n_games = 0  # TRACK NUMBER OF GAMES PLAYED
        self.epsilon = 0  # EXPLORATION RATE
        self.gamma = 0.9  # FOCUSING ON FUTURE REWARDS
        self.memory = Deque(maxlen=MAX_MEMORY)  # EXPERIENCE MEMORY WITH A FIXED SIZE
        self.model = neural_network(11, 256, 3)  # CREATION OF NEURAL NETWORK
        self.trainer = neural_trainer(self.model, lr=LR, gamma=self.gamma)  # CREATION OF TRAINER

    def get_state(self, game):
        # EVALUATES THE CURRENT STATE OF THE GAME AND RETURNS IT
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        # CHECK CURRENT DIRECTION OF THE SNAKE
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # DETERMINE STATE BASED UPON SNAKE'S DIRECTION, COLLISION AND FOOD LOCATION
        state = [
            # DETECT POTENTIAL COLLISIONS
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),

            # CURRENT DIRECTION OF MOVEMENT
            dir_l, dir_r, dir_u, dir_d,

            # FOOD LOCATION RELATIVE TO SNAKES HEAD
            game.food.x < game.head.x,  # FOOD LEFT
            game.food.x > game.head.x,  # FOOD RIGHT
            game.food.y < game.head.y,  # FOOD ABOVE
            game.food.y > game.head.y   # FOOD BELOW
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        # REMEMBERS AND EXPERIENCE AND STORES IT IN DEQUE
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        # TRAINS THE DATA FROM THE BACK CREATED BY LONG TERM MEMORY
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(list(self.memory), BATCH_SIZE)
        else:
            mini_sample = list(self.memory)

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        # TRAINS DATA ON THE MOST RECENT EXPERIENCE
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # DETERMINES NEXT ACTION BASED UPON CURRENT STATE
        self.epsilon = 80 - self.n_games  # DECREASING EXPLORATION FOR BETTER LEARNING
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            # EXPLORATION : CHOOSE A RANODM MOVE
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            # EXPLOITATION : CHOOSE THE BEST ACTION 
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

def train():
    # MAIN TRAINING LOOP
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGame()

    while True:
        # STOP TRAINING ON PRESSING S
        if keyboard.is_pressed('s'):  
            print("Training Stopped.")
            break 

        # GET CURRENT STATE, DECIDE AN ACTION AND MOVE
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # TRAIN THE MODELS SHORT MEMORY (RECENT EXPERIENCE)
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # STORE THE EXPERIENCE IN MEMORY
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # TRAIN THE MODEL WITH LONG TERM MEMORY AND RESET.
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            # UPDATE RECORD AND SCORE AND SAVE THE MODEL
            if score > record:
                record = score
                agent.model.save()

            # DISPLAY TRAINING PROGRESS
            print(f'Game: {agent.n_games}')
            print(f'Score: {score}') 
            print(f'Record: {record}')
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

if __name__ == '__main__':
    train()
