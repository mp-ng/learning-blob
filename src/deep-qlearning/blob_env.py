from blob import Player, Food, Enemy
import numpy as np
from PIL import Image

class BlobEnv:
    SIZE = 10
    RETURN_IMAGES = True
    MOVE_PENALTY = 1
    ENEMY_PENALTY = 300
    FOOD_REWARD = 25
    OBSERVATION_SPACE_VALUES = (SIZE, SIZE, 3)
    ACTION_SPACE_SIZE = 9

    def reset(self):
        self.player = Player(self.SIZE)
        self.food = Food(self.SIZE)

        while self.food == self.player:
            self.food = Food(self.SIZE)

        self.enemy = Enemy(self.SIZE)

        while self.enemy == self.player or self.enemy == self.food:
            self.enemy = Enemy(self.SIZE)

        self.episode_step = 0

        if self.RETURN_IMAGES:
            observation = np.array(self.get_image())
        else:
            observation = (self.player-self.food) + (self.player-self.enemy)

        return observation

    def step(self, choice):
        self.episode_step += 1
        self.player.action(choice)

        if self.RETURN_IMAGES:
            new_observation = np.array(self.get_image())
        else:
            new_observation = (self.player-self.food) + (self.player-self.enemy)

        if self.player == self.enemy:
            reward = -self.ENEMY_PENALTY
        elif self.player == self.food:
            reward = self.FOOD_REWARD
        else:
            reward = -self.MOVE_PENALTY

        done = False
        if reward == self.FOOD_REWARD or reward == -self.ENEMY_PENALTY or self.episode_step >= 200:
            done = True

        return new_observation, reward, done

    def render(self):
        img = self.get_image()
        img = img.resize((300, 300))
        cv2.imshow("image", np.array(img))
        cv2.waitKey(1)

    def get_image(self):
        env = np.zeros((self.SIZE, self.SIZE, 3), dtype=np.uint8)
        env[self.player.x][self.player.y] = self.player.color 
        env[self.food.x][self.food.y] = self.food.color
        env[self.enemy.x][self.enemy.y] = self.enemy.color
        img = Image.fromarray(env, 'RGB')
        return img
        