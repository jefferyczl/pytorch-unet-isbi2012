import random
import collections
import torch
import numpy as np
import numbers
import torchvision.transforms.functional as F
from PIL import Image
_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
}

class ExtCompose(object):
    """Composes several transforms together.
    Args:
        transforms(list of ``Transform`` objects): list of transforms to compose.
    """
    def __init__(self,transforms):
        self.transforms = transforms

    def __call__(self, img, lbl):
        for t in self.transforms:
            img ,lbl = t(img,lbl)
        return img,lbl
    def __repr__(self):
        format_string = self.__class__.__name__+'('
        for t in self.transforms:
            format_string+='\n'
            format_string+='    {0}'.format(t)
        format_string+='\n)'
        return format_string
class ExtElasticTransform(object):
    def __init__(self,alpha=10,sigma=3,p=0.5):
        self.alpah = alpha
        self.sigma=sigma
        self.p = p
    def __call__(self, img,lbl):
        if random.random()>self.p:
            return img,lbl
        # print(img.shape,lbl.shape)
        orig_img_dtype = img.dtype
        img = img.float()
        orig_mask_dtype = lbl.dtype
        lbl = lbl.float()
        combined = torch.cat([img,lbl.unsqueeze(0)],dim=0)
        transformed = self.apply_transform(combined)

        transformed_img = transformed[:img.shape[0]].to(orig_img_dtype)
        transformed_mask = transformed[img.shape[0]:].squeeze(0).to(orig_mask_dtype)
        return transformed_img,transformed_mask
    def apply_transform(self,tensor):
        device = tensor.device
        _,h,w = tensor.shape

        displacement = torch.randn(2,h,w,device=device)*self.alpah

        displacement = F.gaussian_blur(
            displacement.unsqueeze(0),
            kernel_size=int(6*self.sigma)+1,
            sigma=self.sigma
        )[0]
        
        x_grid,y_grid = torch.meshgrid(
            torch.linspace(-1,1,w,device=device),
            torch.linspace(-1,1,h,device=device),
            indexing='xy'
        )
        grid = torch.stack([x_grid,y_grid],dim=-1)

        displacement = displacement.permute(1,2,0)
        displacement[...,0]*=2/w
        displacement[...,1]*=2/h
        grid = grid+displacement

        tensor = tensor.unsqueeze(0)
        grid = grid.unsqueeze(0)

        if tensor.shape[1]>1:
            img_part = torch.nn.functional.grid_sample(
                tensor[:,:-1],
                grid,
                mode='bilinear',
                padding_mode='reflection',
                align_corners=True
            )
            mask_part = torch.nn.functional.grid_sample(
                tensor[:,-1:],
                grid,
                mode='nearest',
                padding_mode='border',
                align_corners=True
            )
            result = torch.cat([img_part,mask_part],dim=1)
        else:
            result = torch.nn.functional.grid_sample(
                tensor,
                grid,
                mode = 'bilinear',
                padding_mode='reflection',
                align_corners=True
            )
        return result[0]
        
class ExtRandomAffine(object):
    def __init__(self,degrees=2,#roate scale
                        translate = (0.05,0.05),
                        scale = 0.95,
                        shear = 2.86,
                        p = 0.8):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.p = p
    def __call__(self, img,lbl):
        assert img.size == lbl.size
        if random.random()<self.p:
            return F.affine(img,self.degrees,self.translate,self.scale,self.shear),F.affine(lbl,self.degrees,self.translate,self.scale,self.shear)
            
        return img,lbl
class ExtRandomAdjustSharpness(object):
    def __init__(self,sharpness_factor=2,
                        p = 0.3):
        self.sharpness_factor = sharpness_factor

        self.p = p
    def __call__(self, img,lbl):
        assert img.size == lbl.size
        if random.random()<self.p:
            return F.adjust_sharpness(img,self.sharpness_factor),lbl
            
        return img,lbl
class ExtAdjustBrightness(object):
    def __init__(self,brightness=0.3,
                        p = 0.3):
        self.brightness = brightness

        self.p = p
    def __call__(self, img,lbl):
        assert img.size == lbl.size
        if random.random()<self.p:
            return F.adjust_brightness(img,self.brightness),lbl
        return img,lbl
            
class ExtGaussianBlur(object):
    def __init__(self,kernal_size,sigma,p):
        self.kernal_size=kernal_size
        self.sigma = sigma
        self.p = p
    def __call__(self, img,lbl):
        assert img.size == lbl.size
        if random.random()<self.p:
            return F.gaussian_blur(img,self.kernal_size,self.sigma),lbl
        return img,lbl

class ExtRandomScale(object):
    def __init__(self,scale_range,interpolation=Image.BILINEAR):
        self.scale_range = scale_range
        self.interpolation = interpolation
    def __call__(self, img,lbl):
        """
        Args:
            img(PIL Image):Image to be scaled.
            lbl(PIL Image):Label to be scaled.
        Returns:
            PIL Image:Rescaled image.
            PIL Image:Rescaled label.
        """
        assert img.size == lbl.size
        scale = random.uniform(self.scale_range[0],self.scale_range[1])
        target_size = (int(img.size[1]*scale),int(img.size[0]*scale))
        return F.resize(img,target_size,self.interpolation),F.resize(lbl,target_size,Image.NEAREST)
    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__+'(size={0},interpolation={1})'.format(self.size,interpolate_str)
    
class ExtRandomCrop(object):
    """Crop the given PIL Image at a random location.
    Args:
        size(sequence or int):Desired output size of the crop. If size is an
            int instead of sequence like (h,w), a square crop (size,size) is 
            made.
        padding (int or sequence,optional): Optional padding on each border of
            the image. Default is 0, i.e no padding. If a sequence of length 4
            is provided, it is used to pad left, top, right, bottom borders 
            respectively.
        pad_if_needed(boolean): It will pad the image if smaller than the desired
            size to avoid raising an exception.
    """
    def __init__(self,size,padding=0,pad_if_needed=False):
        if isinstance(size,numbers.Number):
            self.size = (int(size),int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
    @staticmethod
    def get_params(img,output_size):
        """Get parameters for ``crop`` for a random crop.
        Args:
            img(PIL Image):Image to be cropped.
            output_size(tuple): Expected output size of the crop
        Returns:
            tuple: params (i,j,h,w) to be passed to ``crop`` for random crop.
        """
        w,h = img.size
        th,tw = output_size
        if w == tw and h==th:
            return 0,0,h,w
        i = random.randint(0,h-th)
        j = random.randint(0,w-tw)
        return i,j,th,tw
    
    def __call__(self,img,lbl):
        """
        Args:
            img(PIL Image): Image to be cropped.
            lbl(PIL Image): Label to be cropped.
        Returns:
            PIL Image: Cropped image.
            PIL Image: Cropped label.
        """
        assert img.size == lbl.size,'size of img and lbl should be the same. %s,%s' % (img.size,lbl.size)
        if self.padding>0:
            img = F.pad(img,self.padding)
            lbl = F.pad(lbl,self.padding)
        
        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = F.pad(img,padding = int((1+self.size[1]-img.size[0])/2))
            lbl = F.pad(lbl,padding = int((1+self.size[1]-lbl.size[0])/2))
        
        # pad the height if needed
        if self.pad_if_needed and img.size[1]<self.size[0]:
            img = F.pad(img,padding= int((1+self.size[0]-img.size[1])/2))
            lbl = F.pad(lbl,padding= int((1+self.size[0]-lbl.size[1])/2))
        
        i,j,h,w = self.get_params(img,self.size)

        return F.crop(img,i,j,h,w), F.crop(lbl,i,j,h,w)
    def __repr__(self):
        return self.__class__.__name__+'(size={0},padding={1})'.format(self.size,self.padding)
    
class ExtRandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.
    Args:
        p(float):probability of the image being flipped.Default value is 0.5
    """
    def __init__(self,p = 0.5):
        self.p = p
    def __call__(self,img,lbl):
        """
        Args:
            img(PIL Image): Image to be flipped
        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() <self.p:
            return F.hflip(img),F.hflip(lbl)
        return img,lbl
    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
    
class ExtRandomVerticalFlip(object):
    """Vertical Flip the given PIL Image randomly with a given probability.
    Args:
        p(float):probability of the image being flipped. Default value is 0.5
    """
    def __init__(self,p=0.5):
        self.p = p
    def __call__(self, img,lbl):
        if random.random()<self.p:
            return F.vflip(img),F.vflip(lbl)
        return img,lbl
    def __repr__(self):
        return self.__class__.__name__+'(p={})'.format(self.p)

class ExtToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarrary`` to tensor.
    Converts a PIL Image or numpy.ndarray(H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range[0.0,1.0]
    """
    def __init__(self,normalize=True,target_type='uint8'):
        self.normalize = normalize
        self.target_type = target_type
    def __call__(self,pic,lbl):
        """
        Note that labels will not be normalized to [0,1]
        Args:
            pic(PIL Image or numpy.ndarray):Image to be converted to tensor.
            lbl(PIL Image or numpy.ndarray):Label to be converted to tensor.
        Returns:
            Tensor:Converted image and label
        """
        if self.normalize:
            return F.to_tensor(pic),torch.from_numpy(np.array(lbl,dtype=self.target_type))
        else:
            return torch.from_numpy(np.array(pic,dtype=np.float32).transpose(2,0,1)),torch.from_numpy(np.array(lbl,dtype=self.target_type))
    def __repr__(self):
        return self.__class__.__name__+'()'

class ExtNormalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean:``(M1,...,Mn)`` and std:``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel]-mean[channel])/std[channel]``
    Args:
        mean(sequence): Sequence of means for each channel.
        std(sequence): Sequence of standard deviations for each channel

    """
    def __init__(self,mean,std):
        self.mean = mean
        self.std = std
    def __call__(self,tensor,lbl):
        """
        Args:
            tensor(Tensor):Tensor image of size (C,H,W) to be normalized.
            tensor(Tensor):Tensor of label.A dummpy input for ExtCompose.
        Returns:
            Tensor: Normalized Tensor image.
            Tensor: Unchanged Tensor label.
        """
        return F.normalize(tensor,self.mean,self.std),lbl
    def __repr__(self):
        return self.__class__.__name__+'(mean={0},std={1})'.format(self.mean,self.std)
    
class ExtResize(object):
    """Resize the input PIL Image to the given size.
    Args:
        size(sequence or int):Desired output size. If size is a sequence like
            (h,w),output size will be matched to this.If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height>width,the image will be rescaled to
            (size*height/width,size)
        interpolation (int,optional): Desired interpolation.Default is
            ``PIL.Image.BILINEAR``
    """
    def __init__(self,size,interpolation=Image.BILINEAR):
        assert isinstance(size,int) or (isinstance(size,collections.Iterable) and len(size)==2)
        self.size = size
        self.interpolation = interpolation
    def __call__(self, img,lbl):
        return F.resize(img,self.size,self.interpolation),F.resize(lbl,self.size,Image.NEAREST)
    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__+'(size={0},interpolation={1})'.format(self.size,interpolate_str)

class ExtCenterCrop(object):
    """Crops the given PIL Image at the center.
    Args:
        size(sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h,w), a square crop (size,size) is
            made.
    """
    def __init__(self,size):
        if isinstance(size,numbers.Number):
            self.size = (int(size),int(size))
        else:
            self.size = size
    
    def __call__(self,img,lbl):
        return F.center_crop(img,self.size),F.center_crop(lbl,self.size)
    def __repr__(self):
        return self.__class__.__name__+'(size={0})'.format(self.size)
    