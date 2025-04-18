import PINN.AdamUtilities as au
import Domain.NDDomain as ndd
import Domain.Constants as c

def ndds(skin_layer, tn=1, remove_dim=[]):
    n = ndd.NDDomain(Skin_layer=skin_layer, tn=tn, remove_dim=remove_dim)
    return {key:n for key in au.keys}
            

def T_inf():
    return ndds('Skin_1st', tn=1, remove_dim=[]).ndT(c.Tinf + c.TfWater)


def T_in():
    return ndds('Skin_1st', tn=1, remove_dim=[]).ndT(c.Tin + c.TfWater)