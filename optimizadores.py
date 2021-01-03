from numpy import diag, sqrt, outer

class SGD:
	def __init__(self, learning_rate=0.01):
		self.learning_rate = learning_rate

	def apply_gradients(self, model, epoch):
		model.w += - self.learning_rate * model.g_history[-1]

class SGD_momentum:
	def __init__(self, learning_rate=0.01, alpha=0.1):
		self.learning_rate = learning_rate
		self.alpha = alpha

	def apply_gradients(self, model, epoch):
		try:
			chng = - self.learning_rate * model.g_history[-1] + self.alpha* model.g_history[-2]
		except:
			pass
		finally:
			chng = -self.learning_rate * model.g_history[-1]
		model.w += chng

class AdaGrad:

	def __init__(self, learning_rate=0.05):
		self.learning_rate = learning_rate

	def apply_gradients(self, model, epoch):
		eps = 1e-5
		g = model.g_history[-1]
		diag_G = outer(g,g)[range(len(g)), range(len(g))]
		model.w += - self.learning_rate*((diag_G+eps)**(-0.5))* g

class Adam:
	def __init__(self, learning_rate=0.1):
		self.learning_rate = learning_rate
		self.beta1 = 0.9
		self.beta2 = 0.999
		self.eps = 1e-10
		self.m = 0.
		self.v = 0.

	def apply_gradients(self, model, epoch):
		self.m = self.beta1 * self.m + (1-self.beta1) * model.g_history[-1]
		self.v = self.beta2 * self.v + (1-self.beta2) * model.g_history[-1]**2

		m_var = self.m/(1-self.beta1)
		v_var = self.v/(1-self.beta2)

		model.w -= self.learning_rate * m_var/(sqrt(v_var) + self.eps)
		