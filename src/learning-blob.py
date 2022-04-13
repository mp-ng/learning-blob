from blob import Player, Food, Enemy

import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from matplotlib import style

SIZE = 10
EPISODES = 2000
MOVES_PER_EP = 200
MOVE_PENALTY = 1
ENEMY_PENALTY = 300
FOOD_REWARD = 25
epsilon = 0.9
EPS_DECAY = 0.9998
SHOW_EVERY = 1000

LEARNING_RATE = 0.1
DISCOUNT = 0.95

# Our observation space would be ((x1, y1), (x2, y2))
# where the first tuple is the distance from food
# and the second tuple is the distance from enemy
# so we need all the possible combinations

q_table = {}
for x1 in range(-SIZE + 1, SIZE):
	for y1 in range(-SIZE + 1, SIZE):
		for x2 in range(-SIZE + 1, SIZE):
			for y2 in range(-SIZE + 1, SIZE):
				q_table[((x1, y1),(x2, y2))] = [np.random.uniform(-5, 0) for i in range(4)]

episode_rewards = []

for episode in range(EPISODES):
	player = Player(SIZE)
	food = Food(SIZE)
	enemy = Enemy(SIZE)

	if episode % SHOW_EVERY == 0:
		print(f"on # {episode}, epsilon: {epsilon}")
		print(f"{SHOW_EVERY} ep mean {np.mean(episode_rewards[-SHOW_EVERY:])}")
		show = True
	else:
		show = False

	episode_reward = 0
	for i in range(MOVES_PER_EP):
		obs = (player - food, player - enemy)
		if np.random.random() > epsilon:
			action = np.argmax(q_table[obs])
		else:
			action = np.random.randint(0, 4)

		# Take step
		player.action(action)

		# Calculate reward
		if player.x == enemy.x and player.y == enemy.y:
			reward = -ENEMY_PENALTY
		elif player.x == food.x and player.y == food.y:
			reward = FOOD_REWARD
		else:
			reward = -MOVE_PENALTY

		new_obs = (player - food, player - enemy)
		max_future_q = np.max(q_table[new_obs])
		current_q = q_table[obs][action]

		# If done
		if reward == FOOD_REWARD:
			new_q = FOOD_REWARD
		elif reward == -ENEMY_PENALTY:
			new_q = -ENEMY_PENALTY
		# If not done
		else:
			new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

		q_table[obs][action] = new_q

		if show:
			env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
			env[player.y][player.x] = Player(SIZE).color
			env[food.y][food.x] = Food(SIZE).color
			env[enemy.y][enemy.x] = Enemy(SIZE).color

			img = Image.fromarray(env, "RGB")
			img = img.resize((300, 300), resample=Image.BOX)
			cv2.imshow("", np.array(img))
			if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
				if cv2.waitKey(500) & 0xFF == ord("q"):
					break
			else:
				if cv2.waitKey(1) & 0xFF == ord("q"):
					break
		
		episode_reward += reward
		if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
			break

	episode_rewards.append(episode_reward)
	epsilon *= EPS_DECAY

moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode="valid")

style.use("ggplot")
plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"reward moving-average")
plt.xlabel("episode #")
plt.show()
