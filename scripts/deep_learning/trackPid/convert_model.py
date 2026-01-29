import torch
from models import TrackPIDNetwork2d 

sample = 'contained'

n_features = 3
n_classes = 4 # 0 = muon, 1 = pion, 2 = proton, 3 = kaon
sequence_length = 128
model_depth = 64
n_heads = 4
feed_forward_depth = model_depth*2
n_encoder_layers = 6
n_cross_attention_layers = 2
dropout = 0.2
n_auxillary = 8
my_model = TrackPIDNetwork2d(n_features, n_classes, sequence_length, model_depth, n_heads, feed_forward_depth, n_encoder_layers, n_cross_attention_layers, dropout, n_auxillary)
my_model.load_state_dict(torch.load('best_'+sample+'_model_128_64_4_6_128.pth'))

torchscript_model = torch.jit.script(my_model)
torchscript_model.save('pandora_'+sample+'_track_pid_network_v0.pt')

