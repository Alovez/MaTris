from maze_env import Maze
from RL_brain import DeepQNetwork
from game.matris import Game

def run_maze():
    step = 0
    for episode in range(300):
        # initial observation
        observation = env.reset()

        while True:
            # fresh env
            env.render()

            # RL choose action based on observation
            action = RL.choose_action(observation)

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            RL.store_transition(observation, action, reward, observation_)

            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break
            step += 1

    # end of game
    print('game over')
    env.destroy()


def run_matris():
    step = 0
    for episode in range(10000):
        observation = env.reset()

        while True:
            action = RL.choose_action(observation)

            observation_, reward, done = env.step(action)
            RL.store_transition(observation, action, reward, observation_)
            if (step > 3000) and (step % 20 == 0):
                RL.learn()
            observation = observation_

            if done:
                break
            step += 1


if __name__ == "__main__":
    # maze game
    env = Game()
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=5000,
                      output_graph=True
                      )
    run_matris()
    RL.plot_cost()