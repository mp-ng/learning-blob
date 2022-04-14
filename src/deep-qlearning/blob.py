import numpy as np

class Blob:
	def __init__(self, board_size):
		self.x = np.random.randint(0, board_size)
		self.y = np.random.randint(0, board_size)
		self.board_size = board_size
		self.color = None

	def __str__(self):
		return f"Blob ({self.x}, {self.y})"

	def __sub__(self, other):
		return (self.x - other.x, self.y - other.y)

	def __eq__(self, other):
		return self.x == other.x and self.y == other.y

	def action(self, choice):
		if choice == 0:
			self.move(x=1, y=1)
		elif choice == 1:
			self.move(x=-1, y=-1)
		elif choice == 2:
			self.move(x=-1, y=1)
		elif choice == 3:
			self.move(x=1, y=-1)
		elif choice == 4:
			self.move(x=1, y=0)
		elif choice == 5:
			self.move(x=-1, y=0)
		elif choice == 6:
			self.move(x=0, y=1)
		elif choice == 7:
			self.move(x=0, y=-1)
		elif choice == 8:
			self.move(x=0, y=0)

	def move(self, x=None, y=None):
		if x is None:
			self.x += np.random.randint(-1, 2)
		else:
			self.x += x

		if y is None:
			self.y += np.random.randint(-1, 2)
		else:
			self.y += y

		if self.x < 0:
			self.x = 0
		elif self.x > self.board_size - 1:
			self.x = self.board_size - 1

		if self.y < 0:
			self.y = 0
		elif self.y > self.board_size - 1:
			self.y = self.board_size - 1

class Player(Blob):
	def __init__(self, board_size):
		super().__init__(board_size)
		self.color = (255, 175, 0)

class Food(Blob):
	def __init__(self, board_size):
		super().__init__(board_size)
		self.color = (0, 255, 0)

class Enemy(Blob):
	def __init__(self, board_size):
		super().__init__(board_size)
		self.color = (0, 0, 255)
