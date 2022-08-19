import matplotlib.pyplot as plt
import numpy as np

def plot_sample(logits_mask0,
                propagate_mask0,
                logits_mask1,
                fused_mask,
                uncertainty,
                mask,
                image):
    
    fig = plt.figure(figsize=(20, 80))

    img_count = 0
    bs, num_channels, height, width = image.size()
    for logits0, propagate0, next_logits1, f, u, gt_mask, img in zip(logits_mask0,
                                                                    propagate_mask0,
                                                                    logits_mask1,
                                                                    fused_mask,
                                                                    uncertainty,
                                                                    mask,
                                                                    image):

        plt.subplot(bs, 7, (img_count*7)+ 1)
        data = logits0.detach().cpu().numpy().squeeze()
        data = np.ma.masked_where((0.0 == data), data) #masking the background
        plt.imshow(data, vmin=0, vmax=10)  # convert CHW -> HWC
        plt.title("logits 0")
        plt.axis("off")

        plt.subplot(bs, 7, (img_count*7)+ 6)
        data = gt_mask.detach().cpu().numpy().squeeze()
        data = np.ma.masked_where((0.0 == data), data) #masking the background
        plt.imshow(data, vmin=0, vmax=10) # just squeeze classes dim, because we have only one class
        plt.title("Ground truth")
        plt.axis("off")
        
        plt.subplot(bs, 7, (img_count*7)+ 7)
        img=img.permute(1,2,0)
        plt.imshow(img.detach().cpu().numpy().squeeze()) # just squeeze classes dim, because we have only one class
        plt.title("Image")
        plt.axis("off")

        plt.subplot(bs, 7, (img_count*7)+ 2)
        data = propagate0.detach().cpu().numpy().squeeze()
        data = np.ma.masked_where((0.0 == data), data) #masking the background
        plt.imshow(data, vmin=0, vmax=10) # just squeeze classes dim, because we have only one class
        plt.title("propagate 0")
        plt.axis("off")
      
        plt.subplot(bs, 7, (img_count*7)+ 3)
        data = next_logits1.detach().cpu().numpy().squeeze()
        data = np.ma.masked_where((0.0 == data), data) #masking the background
        plt.imshow(data, vmin=0, vmax=10) # just squeeze classes dim, because we have only one class
        plt.title("next_logits1")
        plt.axis("off")
        
        plt.subplot(bs, 7, (img_count*7)+ 4)
        data = f.detach().cpu().numpy().squeeze()
        data = np.ma.masked_where((0.0 == data), data) #masking the background
        plt.imshow(data, vmin=0, vmax=10) # just squeeze classes dim, because we have only one class
        plt.title("fusion")
        plt.axis("off")
        
        plt.subplot(bs, 7, (img_count*7)+ 5)
        plt.imshow(u.detach().cpu().numpy().squeeze(), vmin=0, vmax=1) # just squeeze classes dim, because we have only one class
        plt.title("uncertainty")
        plt.axis("off")
        img_count = img_count + 1
  
        plt.show()

    return fig
