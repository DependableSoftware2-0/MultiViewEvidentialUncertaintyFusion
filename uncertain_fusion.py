import torch
import numpy as np

class DempsterSchaferCombine(torch.nn.Module):
    '''
    DempsterSchaferCombine will combine 2 dirichlet distribution.
    The assumption is that you will get a batch of predictions.
    The output is shape of [batch*height*width, n_classes]
    '''
    def __init__(self, n_classes):
        super(DempsterSchaferCombine, self).__init__()
        self.n_classes = n_classes
        
    def forward(self, alpha1, alpha2, debug_pixel=0):
        assert (alpha1.ndim == 4) or (alpha1.ndim == 2)
        assert (alpha2.ndim == 4) or (alpha1.ndim == 2)
        assert alpha1.shape == alpha2.shape
        assert alpha1.shape[1] == self.n_classes
        assert alpha2.shape[1] == self.n_classes
        
        reshape_flag = False
        if 4 == alpha1.ndim:
            bs, channels, height, width = alpha1.size()
            # [batch_size,n_classes, height, width] -> [batch_size, height, width, n_classes]
            alpha1 = alpha1.permute(0,2,3,1)
            alpha2 = alpha2.permute(0,2,3,1)
            # [batch_size, height, width, n_classes] -> [batch_size*height*width, n_classes]
            alpha1 = alpha1.reshape(-1, self.n_classes) 
            alpha2 = alpha2.reshape(-1, self.n_classes) 
            reshape_flag = True
        
        #print ("alpha 1 ", debug_pixel, alpha1[debug_pixel])
        #print ("alpha 2 ", debug_pixel, alpha2[debug_pixel])
        
        # Calculate the merger of two DS evidences
        alpha = dict()
        #alpha[0], alpha[1] = alpha1, alpha2
        alpha[0], alpha[1] = alpha1, alpha2
        b, S, E, u = dict(), dict(), dict(), dict()
        for v in range(2):
            S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
            E[v] = alpha[v] - 1
            b[v] = E[v] / (S[v].expand(E[v].shape))
            u[v] = self.n_classes / S[v]

        # b^0 @ b^(0+1)
        bb = torch.bmm(b[0].view(-1, self.n_classes, 1), b[1].view(-1, 1, self.n_classes))
        # b^0 * u^1
        uv1_expand = u[1].expand(b[0].shape)
        bu = torch.mul(b[0], uv1_expand)
        # b^1 * u^0
        uv_expand = u[0].expand(b[0].shape)
        ub = torch.mul(b[1], uv_expand)
        # calculate K
        bb_sum = torch.sum(bb, dim=(1, 2), out=None)
        bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
        # bb_diag1 = torch.diag(torch.mm(b[v], torch.transpose(b[v+1], 0, 1)))
        K = bb_sum - bb_diag

        # calculate b^a
        b_a = (torch.mul(b[0], b[1]) + bu + ub) / ((1 - K).view(-1, 1).expand(b[0].shape))
        # calculate u^a
        u_a = torch.mul(u[0], u[1]) / ((1 - K).view(-1, 1).expand(u[0].shape))
        # test = torch.sum(b_a, dim = 1, keepdim = True) + u_a #Verify programming errors

        # calculate new S
        S_a = self.n_classes / u_a
        # calculate new e_k
        e_a = torch.mul(b_a, S_a.expand(b_a.shape))
        alpha_a = e_a + 1
        
        #print ("alpha_a ", debug_pixel, alpha_a[debug_pixel])
        if reshape_flag:
            # [batch_size*height*width, n_classes] -> [batch_size, height, width, n_classes] 
            alpha_a = alpha_a.reshape(bs, height, width, self.n_classes) 
            # [batch_size, height, width, n_classes] -> [batch_size,n_classes, height, width] 
            alpha_a =  alpha_a.permute(0,3,1,2)  

            
            
        return alpha_a
        
        


class SumUncertainty(torch.nn.Module):
    '''
    SumUncertainty will combine 2 dirichlet distribution.
    The assumption is that you will get a batch of predictions.
    The output is of ndim = 2 or shape of [batch*height*width, n_classes] 
    '''
    def __init__(self, n_classes):
        super(SumUncertainty, self).__init__()
        self.n_classes = n_classes
        
    def forward(self, alpha1, alpha2):
        assert (alpha1.ndim == 4) or (alpha1.ndim == 2)
        assert (alpha2.ndim == 4) or (alpha1.ndim == 2)
        assert alpha1.shape == alpha2.shape
        assert alpha1.shape[1] == self.n_classes
        assert alpha2.shape[1] == self.n_classes
        
        '''
        if 4 == alpha1.ndim:
            # [batch_size,n_classes, height, width] -> [batch_size, height, width, n_classes]
            alpha1 = alpha1.permute(0,2,3,1)
            alhpa2 = alpha2.permute(0,2,3,1)
            # [batch_size, height, width, n_classes] -> [batch_size*height*width, n_classes]
            alpha1 = alpha1.reshape(-1, self.n_classes) 
            alpha2 = alpha2.reshape(-1, self.n_classes) 
         '''
            
        return alpha1 + alpha2
    
class MeanUncertainty(torch.nn.Module):
    '''
    MeanUncertainty will combine 2 dirichlet distribution.
    The assumption is that you will get a batch of predictions.
    The output is of ndim = 2 or shape of [batch*height*width, n_classes] 
    '''
    def __init__(self, n_classes):
        super(MeanUncertainty, self).__init__()
        self.n_classes = n_classes
        
    def forward(self, alpha1, alpha2):
        assert (alpha1.ndim == 4) or (alpha1.ndim == 2)
        assert (alpha2.ndim == 4) or (alpha1.ndim == 2)
        assert alpha1.shape == alpha2.shape
        assert alpha1.shape[1] == self.n_classes
        assert alpha2.shape[1] == self.n_classes
        
        '''
        if 4 == alpha1.ndim:
            # [batch_size,n_classes, height, width] -> [batch_size, height, width, n_classes]
            alpha1 = alpha1.permute(0,2,3,1)
            alhpa2 = alpha2.permute(0,2,3,1)
            # [batch_size, height, width, n_classes] -> [batch_size*height*width, n_classes]
            alpha1 = alpha1.reshape(-1, self.n_classes) 
            alpha2 = alpha2.reshape(-1, self.n_classes) 
        '''
            
        return (alpha1 + alpha2)/2
