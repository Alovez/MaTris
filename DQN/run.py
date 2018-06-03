from maze_env import Maze
from DQN_modified import DeepQNetwork
from game.matris import Game


action_map = ['a', 's', 'd', 'j', 'w']
input_map = {'a':0, 's': 1, 'd': 2, 'w': 4, 'f': 3}

MEMORY_ACCUMULATION = 5000
manually_step = 0

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
    try:
        RL.load_memory()
    except:
        pass
    save_flag = True
    for episode in range(MEMORY_ACCUMULATION * 30000):
        observation = env.reset()

        while True:
            if step < manually_step:
                action = input_map.get(raw_input("action: "), 1)
                # action = RL.choose_action(observation)
            else:
                if save_flag:
                    # RL.save_memory()
                    save_flag = False
            action = RL.choose_action(observation)
            observation_, reward, done = env.step(action_map[action % env.n_actions])
            RL.store_transition(observation, action, reward, observation_)
            if (step > MEMORY_ACCUMULATION) and (step % 50 == 0):
                RL.learn()
            observation = observation_
            print "Step: %s\n Reward: %s\n Action: %s" % (step, reward, action)
            if done:
                break
            step += 1


if __name__ == "__main__":
    # maze game
    env = Game()

    RL = DeepQNetwork(env.n_actions, env.n_features_x[0], env.n_features_y,
                      learning_rate=0.1,
                      reward_decay=0.9,
                      e_greedy=0.1,
                      replace_target_iter=5000,
                      memory_size=MEMORY_ACCUMULATION * 10,
                      output_graph=False
                      )
    run_matris()
    RL.plot_cost()
