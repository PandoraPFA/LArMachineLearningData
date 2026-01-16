import torch
import official_models

nu_classes = 3
trk_classes = 6
shw_classes = 6

my_model = official_models.ConvNeXtV2WithVertexAndNuClass(1, nu_classes, trk_classes, shw_classes, depths=[3, 3, 9, 3], dims=[32, 64, 128, 128], drop_path_rate=0.3, head_init_scale=1.)
my_model.load_state_dict(torch.load('five_hits_with_nu/model_state_dict_large_epoch_37.pt'))

torchscript_model = torch.jit.script(my_model)
torchscript_model.save('pandora_nu_track_shower_counting_network_v0.pt')

