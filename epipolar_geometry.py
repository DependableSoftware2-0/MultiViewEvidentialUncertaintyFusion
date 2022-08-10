import torch
import torch.nn.functional as F


class EpipolarPropagation(torch.nn.Module):
    ''' Epipolar Propagation
    EpipolarPropagation propagates a batch tensor based on the depth 
    and camera parameters.
    Parameters:
        fill_empty_with_ones : The value to fill the empty tensor. 
    ASSUMPTION: Works on on square images because of the clamping
    TODO : Find a way of better clamping rather than magic number 511
    ToDo Remove the loop for filling
    TODO : Assert of intrinsic and image height . need to check datagenerator
    Args:
        K
        Kinv
        height
        width 
        fill_empty_with_ones
        min_depth (float, optional): value used to clamp ``depth``  for
            stability. Default: 0.1 in cm.
    
    
    '''
    def __init__(self, K, Kinv, height, width, fill_empty_with_ones=False, min_depth=0.1):
        super(EpipolarPropagation, self).__init__()
        
        assert K.ndim == 2
        assert Kinv.ndim == 2
        assert height == width #Onlt works for square images, update in dataloader
        
        self.fill_empty_with_ones = fill_empty_with_ones
        self.min_depth = min_depth
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device )
        
        #self.K = torch.Tensor(K).float().to(self.device)
        # good
        self.register_buffer("K", torch.Tensor(K).float())
        self.register_buffer("Kinv", torch.Tensor(Kinv).float())

        #self.Kinv = torch.Tensor(Kinv).float().to(self.device)
        self.height = height
        self.width = width
        
        #checking if cmaera intrinsic is alligned with the image size
        #assert K[0,2]*2 == height
        #commeinting htis as intrinsix image size is 511 and actual image is 512 need to debug
        
        #Getting index of each pixel of image in [n,3, height*width] shape device=self.device
        #self.grid_x, self.grid_y = torch.meshgrid(torch.arange(height, device=self.device), 
        #                                torch.arange(width, device=self.device), indexing='ij')
        grid_x, grid_y = torch.meshgrid(torch.arange(height), 
                                        torch.arange(width), indexing='ij')      
        grid_x = torch.flatten(grid_x, start_dim=0)
        grid_y = torch.flatten(grid_y, start_dim=0)
        
        self.register_buffer("grid_x", grid_x) 
        self.register_buffer("grid_y", grid_y)
        ## Stacking to make the matrix 3 x points
        #self.index = torch.vstack((grid_x, grid_y, torch.ones_like(grid_y))).float()
        self.register_buffer("index", torch.vstack((grid_x, grid_y, torch.ones_like(grid_y))).float())
        
    def forward(self, image, depth, T, R):
        assert torch.is_tensor(image) == True
        assert torch.is_tensor(depth) == True
        assert torch.is_tensor(T) == True
        assert torch.is_tensor(R) == True
        assert image.ndim == 4
        assert depth.ndim == 3
                
        #Getting depth and flatenning it and making it [n, 1, height*width] shape
        flattened_depth = torch.flatten(depth, start_dim=1)
        flattened_depth = torch.unsqueeze(flattened_depth, dim=1).float().to(self.device)

        #Getting Transformation and Rotation from previous frame
        T = T.float().to(self.device)
        R = R.float().to(self.device)
        
        # Entries of var must be non-negative
        if torch.any(flattened_depth < 0):
            raise ValueError("Depth has negative entry/entries")

        # Clamp for stability
        flattened_depth=torch.clamp(flattened_depth, min=self.min_depth)
        
        #Epipolar Geometry Equations
        x = torch.matmul(torch.matmul(torch.matmul(self.K, R), self.Kinv), self.index)
        if (torch.isnan(x).any()):
            raise ValueError("Matrix mutiplication K, R , Kinv and index contains Nan")      
        y = torch.matmul(self.K,torch.div(T, flattened_depth)) 
        if (torch.isnan(y).any()):
            raise ValueError("Matrix mutiplication K, T and depth  contains Nan") 
        
        #When adding the sum some values go nan replace them with zero
        transposed_index = x + y 
        transposed_index[transposed_index==0.0] = 0.0001
                
        ## Dividing Last column for each image
        transposed_index = torch.div(transposed_index, transposed_index[:,2,:].unsqueeze(1))
        
        if (torch.isnan(transposed_index).any()):
            print ("K ", self.K)
            print ("Kinv", self.Kinv)
            #print ("index", self.index)
            print ("depth ", torch.min(flattened_depth))
            x = torch.matmul(self.K,torch.div(T, flattened_depth))
            y =  torch.matmul(torch.matmul(torch.matmul(self.K, R), self.Kinv), self.index)
            z = torch.nan_to_num(x+y, nan=0.1)
            print ("x + y what ", z.min(), torch.isnan(z.view(-1)).sum() )
            print ("last row z ", z[:,2,:].min(), torch.isnan(z[:,2,:]).sum()) 
            z = torch.div(z, z[:,2,:].unsqueeze(1))
            idx = torch.isnan(z)
            print ("value of nan at x and y ", x[idx], y[idx])
            print ("after divide  row ", z.min(), torch.isnan(z).sum() ) 
            print ("transposed_index ", torch.min(transposed_index), torch.isnan(transposed_index.view(-1)).sum())

            raise ValueError("Transposed index contains Nan")
            
        ## Clamping index ,  imagesize subtracting 1
        transposed_index = torch.clamp(transposed_index, min=0, max=self.height-1).long() # 
        #Projecting pixels from previous frame based on the transposed index
        if self.fill_empty_with_ones:
            projected_tensor = torch.ones_like(image, device=self.device)
        else:
            projected_tensor = torch.zeros_like(image, device=self.device)

        image = image.to(self.device)
        #ToDo can this be done without for loop
        for i,p in enumerate(transposed_index):
            projected_tensor[i,:,p[0],p[1]] = image[i,:,self.grid_x, self.grid_y]

        projected_tensor = projected_tensor.float()
        #m = torch.nn.MaxPool2d(2)
        projected_tensor = F.max_pool2d(projected_tensor, kernel_size=2)

        projected_tensor = F.interpolate(projected_tensor, scale_factor=2 )
        
        return projected_tensor
    
