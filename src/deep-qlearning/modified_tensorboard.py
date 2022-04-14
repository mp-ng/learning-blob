from tensorflow.keras.callbacks import TensorBoard

class ModifiedTensorBoard(TensorBoard):

	# Override: Set initial step and writer (one log file for all .fit() calls)
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.step = 1
		self.writer = tf.summary.FileWriter(self.log_dir)

	# Override: Stop creating default log writer
	def set_model(self, model):
		pass

	# Override: Save logs with our step number and not from 0th step every time
	def on_epoch_end(self, epoch, logs=None):
		self.update

	# Override: Train for one batch only, no need to save anything at epoch end
	def on_batch_end(self, batch, logs=None):
		pass

	# Override: Won't close writer
	def on_train_end(self, _):
		pass

	def update_stats(self, **stats):
		self._write_logs(stats, self.step)
