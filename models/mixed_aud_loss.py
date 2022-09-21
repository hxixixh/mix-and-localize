import torch
import torch.nn.functional as F


class MixAudSIS1FeatHardCycleLoss(): 
    def __init__(self, num_aud_node, cycle_temp): 
        self.num_aud_node = num_aud_node
        self.cycle_temp = cycle_temp
    
    def compute_loss(self, output, max_mode=False, temp=0.1): 
        diags = {}

        loss, diags_ = self.compute_cycle_loss(output, max_mode=max_mode, temp=self.cycle_temp)
        diags.update(diags_)

        diags.update({"total xent": loss.detach()})

        return loss, diags

    def compute_cycle_loss(self, output, max_mode=False, temp=0.07):
        av_outputs, av_map, v, a_s = output
        a_ = torch.stack(a_s, dim=1).squeeze().reshape(-1, 128, 1)
        B = v.shape[0]
        a_1 = a_[:B]    
        a_2 = a_[B:]

        feat_img = v.reshape(B // self.num_aud_node, self.num_aud_node, 128, 14*14).permute(0, 1, 3, 2)     # [B//2, 2, 14*14, 128]
        feat_aud_1 = a_1.reshape(B // self.num_aud_node, self.num_aud_node, 128).permute(0, 2, 1)     # [B//2, 128, num_aud_nodes]
        feat_aud_2 = a_2.reshape(B // self.num_aud_node, self.num_aud_node, 128).permute(0, 2, 1)     # [B//2, 128, num_aud_nodes]

        A_IS = torch.bmm(feat_img.reshape(B // self.num_aud_node, self.num_aud_node*14*14, 128), feat_aud_1)  # [B//2, 2*14*14, num_aud_nodes]
        A_IS = A_IS.reshape(B // self.num_aud_node, self.num_aud_node, -1, self.num_aud_node).max(dim=2)[0]
        
        A_SI = torch.bmm(feat_aud_2.permute(0, 2, 1), feat_img.reshape(B // self.num_aud_node, self.num_aud_node*14*14, 128).permute(0, 2, 1))
        A_SI = A_SI.reshape(B // self.num_aud_node, self.num_aud_node, self.num_aud_node, -1).max(dim=-1)[0]

        A_IS = torch.softmax(A_IS/self.cycle_temp, dim=-1)
        A_SI = torch.softmax(A_SI/self.cycle_temp, dim=-1)    

        A_SIS = torch.bmm(A_SI, A_IS).reshape(B // self.num_aud_node, self.num_aud_node, self.num_aud_node) # [B//2, 2, 2]

        labels = torch.arange(A_SIS.shape[1]).repeat(A_SIS.shape[0]).to(A_SIS)
        loss = torch.mean(F.nll_loss(torch.log(A_SIS.reshape(B//self.num_aud_node*self.num_aud_node, -1)), labels.long(), reduction='none'))

        acc_ASIS = (torch.argmax(A_SIS.reshape(B//self.num_aud_node*self.num_aud_node, -1), dim=-1) == labels).float().mean()

        diags = dict()
        diags.update({"ASIS xent": loss.detach(),
                        "ASIS acc": acc_ASIS})

        return loss, diags