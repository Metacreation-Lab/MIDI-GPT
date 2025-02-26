import torch

def standard_loss(self, model, inputs, return_outputs=False):
	outputs = model(**inputs)
	if self.args.past_index >= 0:
		self._past = outputs[self.args.past_index]
	loss = outputs[0].mean()
	print("loss : ", loss)
	return (loss,outputs) if return_outputs else loss

def hinge_cost(m, a, b):
	dist = m - torch.sqrt(torch.sum((a - b)**2, axis=1))
	return torch.mean(torch.clamp(dist,0,float('inf'))**2)

def sim_metric_loss(self, model, inputs, return_outputs=False):

	# single pass version
	batch_size = len(inputs["input_ids"])//4
	outputs = model(**inputs)
	x_p = outputs[:batch_size]
	x_n = outputs[batch_size:2*batch_size]
	y_p = outputs[2*batch_size:3*batch_size]
	y_n = outputs[3*batch_size:]

	model_attr = model
	if isinstance(model, torch.nn.DataParallel):
		model_attr = model.module

	cost_p = torch.mean(torch.sum((x_p - y_p)**2, axis=1))
	cost_n = model_attr.negative_importance*hinge_cost(
			model_attr.negative_threshold, x_n, y_n)
	cost_e = model_attr.entropy_importance*torch.mean(
			torch.sum(x_p**2, axis=1) + torch.sum(y_p**2, axis=1))
	loss = cost_p + cost_n + cost_e

	print(loss)
	return (loss,None) if return_outputs else loss
