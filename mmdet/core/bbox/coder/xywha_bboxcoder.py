import mmcv
import numpy as np
import torch
from mmdet.core.bbox.coder.base_bbox_coder import BaseBBoxCoder

from ..builder import BBOX_CODERS
from ..transforms import norm_angle

@BBOX_CODERS.register_module()
class XYWHABBoxCoder(BaseBBoxCoder):
    """
    新建一个子类 XYWHABBoxCoder,继承父类BaseBBoxCoder
    用于将坐标 (xc, yc, w, h, a)编码成(dx, dy, dw, dh, da)
    以及将(dx, dy, dw, dh, da)解码成(xc, yc, w, h, a)
    """
    def __init__(self,
                 target_means=(0., 0., 0., 0., 0.),
                 target_stds=(1., 1., 1., 1., 1.),
                 angle_range='oc',
                 norm_factor=None,
                 edge_swap=False,
                 proj_xy=False,
                 add_ctr_clamp=False,
                 ctr_clamp=32):
        super(BaseBBoxCoder, self).__init__()
        self.means = target_means
        self.stds = target_stds
        self.add_ctr_clamp = add_ctr_clamp
        self.ctr_clamp = ctr_clamp
        self.angle_range = angle_range
        self.norm_factor = norm_factor
        self.edge_swap = edge_swap
        self.proj_xy = proj_xy
    
    def encode(self, bboxes, gt_bboxes):
        """
        得到参数: box regression transformation deltas
        """
        assert bboxes.size(0) == gt_bboxes.size(0)
        assert bboxes.size(-1) == 5
        assert gt_bboxes.size(-1) == 5
        if self.angle_range in ['oc', 'le135', 'le90']:
            return bbox2delta(bboxes, gt_bboxes, self.means, self.stds,
                              self.angle_range, self.norm_factor,
                              self.edge_swap, self.proj_xy)
        else:
            raise NotImplementedError
    
    # 用于解码的函数
    def decode(self,
               bboxes,
               pred_bboxes,
               max_shape=None,
               wh_ratio_clip=16 / 1000):
        """
        将pred_bboxes转换为boxes
        """
        assert pred_bboxes.size(0) == bboxes.size(0)
        if self.angle_range in ['oc', 'le135', 'le90']:
            return delta2bbox(bboxes, pred_bboxes, self.means, self.stds,
                              max_shape, wh_ratio_clip, self.add_ctr_clamp,
                              self.ctr_clamp, self.angle_range,
                              self.norm_factor, self.edge_swap, self.proj_xy)
        else:
            raise NotImplementedError

# 编码解码过程中需要用到的转换的函数
def bbox2delta(proposals,
               gt,
               means=(0., 0., 0., 0., 0.),
               stds=(1., 1., 1., 1., 1.),
               angle_range='oc',
               norm_factor=None,
               edge_swap=False,
               proj_xy=False):

    assert proposals.size() == gt.size()
    proposals = proposals.float()
    gt = gt.float()
    px, py, pw, ph, pa = proposals.unbind(dim=-1)
    gx, gy, gw, gh, ga = gt.unbind(dim=-1)

    if proj_xy:
        dx = (torch.cos(pa) * (gx - px) + torch.sin(pa) * (gy - py)) / pw
        dy = (-torch.sin(pa) * (gx - px) + torch.cos(pa) * (gy - py)) / ph
    else:
        dx = (gx - px) / pw
        dy = (gy - py) / ph

    if edge_swap:
        dtheta1 = norm_angle(ga - pa, angle_range)
        dtheta2 = norm_angle(ga - pa + np.pi / 2, angle_range)
        abs_dtheta1 = torch.abs(dtheta1)
        abs_dtheta2 = torch.abs(dtheta2)
        gw_regular = torch.where(abs_dtheta1 < abs_dtheta2, gw, gh)
        gh_regular = torch.where(abs_dtheta1 < abs_dtheta2, gh, gw)
        da = torch.where(abs_dtheta1 < abs_dtheta2, dtheta1, dtheta2)
        dw = torch.log(gw_regular / pw)
        dh = torch.log(gh_regular / ph)
    else:
        da = norm_angle(ga - pa, angle_range)
        dw = torch.log(gw / pw)
        dh = torch.log(gh / ph)

    if norm_factor:
        da /= norm_factor * np.pi

    deltas = torch.stack([dx, dy, dw, dh, da], dim=-1)
    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)
    return deltas


@mmcv.jit(coderize=True)
def delta2bbox(rois,
               deltas,
               means=(0., 0., 0., 0., 0.),
               stds=(1., 1., 1., 1., 1.),
               max_shape=None,
               wh_ratio_clip=16 / 1000,
               add_ctr_clamp=False,
               ctr_clamp=32,
               angle_range='oc',
               norm_factor=None,
               edge_swap=False,
               proj_xy=False):
    
    means = deltas.new_tensor(means).view(1, -1).repeat(1, deltas.size(1) // 5)
    stds = deltas.new_tensor(stds).view(1, -1).repeat(1, deltas.size(1) // 5)
    denorm_deltas = deltas * stds + means
    dx = denorm_deltas[:, 0::5]
    dy = denorm_deltas[:, 1::5]
    dw = denorm_deltas[:, 2::5]
    dh = denorm_deltas[:, 3::5]
    da = denorm_deltas[:, 4::5]
    if norm_factor:
        da *= norm_factor * np.pi
    # Compute center of each roi
    px = rois[:, 0].unsqueeze(1).expand_as(dx)
    py = rois[:, 1].unsqueeze(1).expand_as(dy)
    # Compute width/height of each roi
    pw = rois[:, 2].unsqueeze(1).expand_as(dw)
    ph = rois[:, 3].unsqueeze(1).expand_as(dh)
    # Compute rotated angle of each roi
    pa = rois[:, 4].unsqueeze(1).expand_as(da)
    dx_width = pw * dx
    dy_height = ph * dy
    max_ratio = np.abs(np.log(wh_ratio_clip))
    if add_ctr_clamp:
        dx_width = torch.clamp(dx_width, max=ctr_clamp, min=-ctr_clamp)
        dy_height = torch.clamp(dy_height, max=ctr_clamp, min=-ctr_clamp)
        dw = torch.clamp(dw, max=max_ratio)
        dh = torch.clamp(dh, max=max_ratio)
    else:
        dw = dw.clamp(min=-max_ratio, max=max_ratio)
        dh = dh.clamp(min=-max_ratio, max=max_ratio)
    # Use exp(network energy) to enlarge/shrink each roi
    gw = pw * dw.exp()
    gh = ph * dh.exp()
    # Use network energy to shift the center of each roi
    if proj_xy:
        gx = dx * pw * torch.cos(pa) - dy * ph * torch.sin(pa) + px
        gy = dx * pw * torch.sin(pa) + dy * ph * torch.cos(pa) + py
    else:
        gx = px + dx_width
        gy = py + dy_height
    # Compute angle
    ga = norm_angle(pa + da, angle_range)
    if max_shape is not None:
        gx = gx.clamp(min=0, max=max_shape[1] - 1)
        gy = gy.clamp(min=0, max=max_shape[0] - 1)

    if edge_swap:
        w_regular = torch.where(gw > gh, gw, gh)
        h_regular = torch.where(gw > gh, gh, gw)
        theta_regular = torch.where(gw > gh, ga, ga + np.pi / 2)
        theta_regular = norm_angle(theta_regular, angle_range)
        return torch.stack([gx, gy, w_regular, h_regular, theta_regular],
                           dim=-1).view_as(deltas)
    else:
        return torch.stack([gx, gy, gw, gh, ga], dim=-1).view(deltas.size())
