from transformers import *
import torch.nn as nn
import torch

class GPT2Encoder(GPT2PreTrainedModel):
	def __init__(self, config):
		super().__init__(config)
		self.transformer = GPT2Model(config)
		self.score = nn.Linear(config.n_embd, 128, bias=False)
		self.init_weights()
		self.model_parallel = False
		self.device_map = None
		
		negative_importance = torch.tensor(5.33).float()
		negative_threshold = torch.tensor(4.).float()
		entropy_importance = torch.tensor(0.05).float()
		self.register_buffer('negative_importance', negative_importance)
		self.register_buffer('negative_threshold', negative_threshold)
		self.register_buffer('entropy_importance', entropy_importance)

	def forward(
		self,
		input_ids=None,
		past_key_values=None,
		attention_mask=None,
		token_type_ids=None,
		position_ids=None,
		head_mask=None,
		inputs_embeds=None,
		labels=None,
		use_cache=None,
		output_attentions=None,
		output_hidden_states=None,
		return_dict=None,
		sequence_lengths=None,
	):
		assert sequence_lengths is not None
		outputs = self.transformer(
			input_ids,
			past_key_values=past_key_values,
			attention_mask=attention_mask,
			token_type_ids=token_type_ids,
			position_ids=position_ids,
			head_mask=head_mask,
			inputs_embeds=inputs_embeds,
			use_cache=use_cache,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)
		hidden_states = outputs[0]
		logits = self.score(hidden_states)
		return logits[range(input_ids.shape[0]),sequence_lengths-1]

class GPT2LMHeadModelContConfig(GPT2Config):
	def __init__(
		self,
		n_control_embd=64,
		n_control_dim=3,
		**kwargs
	):
		super().__init__(**kwargs)
		self.n_control_embd = n_control_embd
		self.n_control_dim = n_control_dim

class GPT2LMHeadModelCont(GPT2LMHeadModel):
	def __init__(self, config):
		super().__init__(config)
		token_embd = config.n_embd - config.n_control_embd
		self.wte = nn.Embedding(config.vocab_size, token_embd)
		self.ctrle = nn.Linear(config.n_control_dim, config.n_control_embd)

	def forward(
		self,
		input_ids=None,
		control_ids=None,
		past_key_values=None,
		attention_mask=None,
		token_type_ids=None,
		position_ids=None,
		head_mask=None,
		inputs_embeds=None,
		labels=None,
		use_cache=None,
		output_attentions=None,
		output_hidden_states=None,
		return_dict=None,
		sequence_lengths=None
	):
		shape = control_ids.shape
		input_shape = (shape[0]*shape[1],shape[2])
		output_shape = (shape[0],shape[1],self.config.n_control_embd)
		control_embd = self.ctrle(torch.reshape(control_ids, input_shape))
		control_embd = torch.reshape(control_embd, output_shape)
		token_embd = self.wte(input_ids)
		inputs_embeds = torch.cat([token_embd,control_embd], axis=-1)
		return super().forward(
			past_key_values=past_key_values,
			attention_mask=attention_mask,
			inputs_embeds=inputs_embeds,
			labels=labels
		)

if __name__ == "__main__":

	batch_size = 3

	config = GPT2Config().from_json_file("config/gpt2_tiny.json")
	
	#model = GPT2LMHeadModelCont(config)
	model = GPT2Encoder(config)

	kwargs = {
		"input_ids" : torch.randint(config.vocab_size, size=(batch_size,100)),
		"labels" : torch.randint(config.vocab_size, size=(batch_size,100)),
		#"control_ids" : torch.randint(CONTROL_VOCAB_SIZE, size=(1,100))
		"sequence_lengths" : [99,90,90]
	}

	out = model.forward(**kwargs)
	print(out.shape)
