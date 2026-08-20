from torch.utils.data import Dataset
import torch
import numpy as np
import nibabel as nib
import cv2
from pathlib import Path


def totensor(array):
    """Convert a float HxW or HxWxC NumPy array to a CHW tensor."""
    array = np.asarray(array)
    if array.ndim == 2:
        array = array[None, ...]
    elif array.ndim == 3:
        array = array.transpose(2, 0, 1)
    else:
        raise ValueError(f"Expected a 2D or 3D array, got shape {array.shape}")
    return torch.from_numpy(np.ascontiguousarray(array))


def pad(img, x):  # sourcery skip: avoid-builtin-shadow
    size = img.shape
    if len(size)==4:
        h, w, c, t = size
        top, bottom, left, right = (x - h)//2, (x - h + 1)//2, (x - w)//2, (x - w + 1)//2
        list = [
            cv2.copyMakeBorder(
                img[:,:,:,i], top, bottom, left, right, cv2.BORDER_CONSTANT, value=0
            )
            for i in range(t)
        ]
        return np.stack(list, axis = 3)
    else:
        h, w, c= size
        top, bottom, left, right = (x - h)//2, (x - h + 1)//2, (x - w)//2, (x - w + 1)//2
        return cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0
        )

def resize_worker(lr_file):
    load_lr = nib.load(lr_file)
    
    head_affine = (load_lr.header, load_lr.affine)
    img_lr = load_lr.get_fdata()
    
    ii = str(lr_file.parent)
    b0_index_l = [0]
    b1000_index = list(range(1, 31))
    #b0_index_l = [0, 7, 14, 21]
    #b1000_index = list(set(range(28)) - set(b0_index_l))
    #b0_index_l = [0, 31]
    #b1000_index = list(set(range(32)) - set(b0_index_l))
    #b0_index_l = [0, 4, 15, 28]
    #b1000_index = list(set(range(36)) - set(b0_index_l))
    #b0_index_l = [0,1,5,16,27,38]
    #b1000_index = list(set(range(44)) - set(b0_index_l)) #lht changed
    #b0_index_l = [27,50]
    #b1000_index = [44,45,46,47,48,49,51,52,53,54]
    # mask = nib.load(f'{ii}/mask.nii').get_fdata()
    # img_T1 =  nib.load(f'{ii}/T1.nii').get_fdata()
    mask = nib.load(f'{ii}/T1_brain_mask.nii.gz').get_fdata()
    img_T1 =  nib.load(f'{ii}/T1_brain.nii.gz').get_fdata()

    dim_l = load_lr.header['pixdim'][1]
    dim_t = nib.load(f'{ii}/T1_brain.nii.gz').header['pixdim'][1]
    fac1 = dim_l/dim_t
    fac2 = load_lr.header['pixdim'][2]/nib.load(f'{ii}/T1_brain.nii.gz').header['pixdim'][2]
    print(img_lr.shape)
    
    # img_T1 = (img_T1.T * mask.T).T

    # rescale in axial
    cv_limit = 512
    hh, ww, c, t = img_lr.shape 
    flat_lr = img_lr.reshape(hh, ww, -1)
    ir = np.concatenate([cv2.resize(flat_lr[:, :, i:min(i + cv_limit, flat_lr.shape[2])],dsize=None, fx=fac2,fy=fac1, interpolation=cv2.INTER_CUBIC)
                            for i in range(0, flat_lr.shape[2], cv_limit)], axis=2)
    img_lr = ir.reshape(int(hh*fac1), int(ww*fac2), c, t)
    # img_lr = pad(img_lr, 256)

    # rescale in sag
    # lr = img_lr.transpose((2,1,0,3))
    # lr = np.flipud(lr)
    # hh, ww, c, t = lr.shape
    # flat_lr = lr.reshape(hh, ww, -1)
    # ir = np.concatenate([cv2.resize(flat_lr[:, :, i:min(i + cv_limit, flat_lr.shape[2])],dsize=None, fx=1,fy=2, interpolation=cv2.INTER_CUBIC)
    #                         for i in range(0, flat_lr.shape[2], cv_limit)], axis=2)
    # ir = ir.reshape(hh*2, ww, c, t)
    # lr = np.flipud(ir)
    # lr = lr.transpose((2,1,0,3))

    # --------- rescale for mask------------
    # rescale in axial
    # mask = cv2.resize(mask, dsize=None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # rescale in sag
    # mask = mask.transpose((2,1,0))
    # mask = np.flipud(mask)
    # mask = cv2.resize(mask, dsize=None, fx=1, fy=2, interpolation=cv2.INTER_CUBIC)
    
    # mask = np.flipud(mask)
    # mask = mask.transpose((2,1,0))
    # lr = lr[:,range(1,ww*factor),:,:]
    
    # hh, ww, c, t = lr.shape
    # flat_lr = lr.reshape(hh, ww, -1)
    # lr = np.concatenate([cv2.resize(flat_lr[:, :, i:min(i + cv_limit, flat_lr.shape[2])],dsize=None, fx=fac,fy=fac, interpolation=cv2.INTER_CUBIC)
    #                         for i in range(0, flat_lr.shape[2], cv_limit)], axis=2)
    # lr = lr.reshape(int(hh*fac), int(ww*fac), c, t)
    # mask = cv2.resize(mask, dsize=None, fx=fac, fy=fac, interpolation=cv2.INTER_CUBIC)
    # mask = np.where(mask>0.4, 1, 0)
    #-------------------------------------------

    
    # lr = lr.transpose((1,0,2,3))
    # lr = lr[::-1]
    # mask = np.flipud(mask)
    # mask = mask.transpose((1,0,2))
    # mask = mask[::-1]
    # T1img = np.flipud(img_T1)
    # T1img = T1img.transpose((1,0,2))
    # T1img = T1img[::-1]
    
    # ir = pad(lr, 256)
    # mask = pad(mask, 256)
    # # print(ir.shape, mask.shape)
    # ir = (ir.T * mask.T).T
    # T1img = pad(T1img, 256)
    # T1img =(T1img.T * mask.T).T

    # lr = pad(img_lr, 256)
    # mask = pad(mask, 256)
    # img_T1 = pad(img_T1, 256)
    # # # if direction=='sag':
    # lr = lr.transpose((2,1,0,3))
    # lr = np.flipud(lr)
    # # mask = np.flipud(mask)
    # mask = mask.transpose((2,1,0))
    # mask = np.flipud(mask)
    # # T1img = np.flipud(img_T1)
    # T1img = img_T1.transpose((2,1,0))
    # T1img = np.flipud(T1img)
    ir = np.flipud(img_lr)
    ir = ir.transpose(0,2,1,3)
    ir = np.flipud(ir)
    T1img = np.flipud(img_T1)
    T1img = T1img.transpose(0,2,1)
    T1img = np.flipud(T1img)
    mask = np.flipud(mask)
    mask = mask.transpose(0,2,1)
    mask= np.flipud(mask)
    # ir = pad(lr, 256)
    # # print(mask.shape)
    # mask = pad(mask, 256)
    # # #--------for test---------
    # ir = (ir.T * mask.T).T
    # # #-------------------------
    # T1img = pad(T1img, 256)
    # T1img =(T1img.T * mask.T).T
    ir = pad(ir, 256)
    T1img = pad(T1img, 256)
    mask = pad(mask, 256)
    maxn_i = []
    for b in range(t):
        maxx = ir[:,:,:,b].max()
        hist, bins = np.histogram(ir[:,:,:,b], bins=1000, range=(0.1, ir[:,:,:,b].max()))
        for index in range(len(hist)):
            if hist[:len(hist)-index].sum()/hist.sum() < 0.997:
                # print(f'index = {len(hist)-index}, {hist[:len(hist)-index].sum()}/{hist.sum()}, maxn = {bins[len(bins)-index]}/{maxn}')
                maxx = bins[len(bins)-index]
                break
        maxn_i.append(maxx)
        ir[:,:,:,b] = ir[:,:,:,b]/maxx

    maxT = T1img.max()
    histT, binsT = np.histogram(T1img, bins=1000, range=(0.1, T1img.max()))
    for indexT in range(len(histT)):
        if histT[:len(histT)-indexT].sum()/histT.sum() < 0.997:
            # print(f'index = {len(hist)-index}, {hist[:len(hist)-index].sum()}/{hist.sum()}, maxn = {bins[len(bins)-index]}/{maxn}')
            maxT = binsT[len(binsT)-indexT]
            break
    T1img = T1img/maxT
    # T1img =(T1img.T * mask.T).T
    maxn_i= np.array(maxn_i)
    return [ir, T1img, mask], [b0_index_l, b1000_index], [maxn_i, maxT], head_affine
    


def prepare(lr_path):
    print('prepare start')
    lr_path = Path(lr_path)
    imgs, bindexs, maxn, head_affine = resize_worker(lr_path)
    print('prepare finished')
    return imgs, bindexs, maxn, head_affine


class lrgraceset(Dataset):
    def __init__(self, lr_path):
        imgs, indexs, maxn, head_affine = prepare(lr_path)
        ir, T1img, mask = imgs
        self.head_affine = head_affine
        b0_index_l, b1000_index = indexs
        self.maxn = maxn
        self.b1000_index = b1000_index
        self.b0_index_l = b0_index_l
        
        self.ir = ir
        self.T1img = T1img
        self.mask = mask

        # self.direc('sag')
    
    # def changeir

    def direc(self, direction):
        """Arrange data as ``(plane_axis_0, plane_axis_1, slice_axis)``.

        ``resize_worker`` stores its padded base volume as ``(x, z, y)``.
        These explicit permutations make the public direction names anatomical:
        axial -> (x, y), cor -> (x, z), sag -> (y, z).
        """
        direction = direction.lower()
        if direction == 'axial':
            # (x, z, y) -> (x, y, z); slice over z.
            axes_3d = (0, 2, 1)
            axes_4d = (0, 2, 1, 3)
        elif direction == 'cor':
            # Already (x, z, y); slice over y.
            axes_3d = (0, 1, 2)
            axes_4d = (0, 1, 2, 3)
        elif direction == 'sag':
            # (x, z, y) -> (y, z, x); slice over x.
            axes_3d = (2, 1, 0)
            axes_4d = (2, 1, 0, 3)
        else:
            raise ValueError("direction 必须是 axial、cor 或 sag")

        self.ir = np.ascontiguousarray(self.ir.transpose(axes_4d))
        self.T1img = np.ascontiguousarray(self.T1img.transpose(axes_3d))
        self.mask = np.ascontiguousarray(self.mask.transpose(axes_3d))
        self.direction = direction
        self.b0_index_l = [0]
        print(f'direction={direction}, network volume shape={self.ir.shape}')

        _, _, c, _ = self.ir.shape
        print(self.ir.shape)
        self.data_len = c*len(self.b1000_index)
        data_dict = {}
        lr_data = []
        t1_data = []
        mask_data = []
        for s in range(self.data_len):
            slice = s//len(self.b1000_index)
            b = s%len(self.b1000_index)
            if len(self.b0_index_l)>1:
                b0_l = totensor(self.ir[:,:,slice,self.b0_index_l].mean(2)).to(torch.float32).clamp_(0,1)
            else:
                b0_l = totensor(self.ir[:,:,slice,self.b0_index_l]).to(torch.float32).clamp_(0,1)
            b1000_l = totensor(self.ir[:,:,slice,self.b1000_index[b]]).to(torch.float32).clamp_(0,1)
            lr_data.append(torch.cat([b0_l, b1000_l], dim=0))
            t1_data.append(totensor(self.T1img[:,:,slice]).to(torch.float32))
            mask_data.append(totensor(self.mask[:,:,slice]).to(torch.float32))

        self.lr_data = lr_data
        self.t1_data = t1_data
        self.mask_data = mask_data

    @staticmethod
    def restore_volume(volume, direction):
        """Restore a sampled direction-layout volume to canonical ``(x,y,z)``."""
        direction = direction.lower()
        if direction == 'axial':
            axes = (0, 1, 2, 3)
        elif direction == 'cor':
            axes = (0, 2, 1, 3)
        elif direction == 'sag':
            axes = (2, 0, 1, 3)
        else:
            raise ValueError("direction 必须是 axial、cor 或 sag")
        return np.ascontiguousarray(volume.transpose(axes))

    def __len__(self):
        return self.data_len
    
    def __getitem__(self, index):
        return self.lr_data[index], self.t1_data[index], self.mask_data[index]
