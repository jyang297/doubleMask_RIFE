import torch
import torch.nn as nn
import torch.nn.functional as F
from model.warplayer import warp
from model.refine import *


class Student_Clean_Warp_IFBlock(nn.Module):
    def __init__(self, in_planes=6, c=64):
        super().__init__()
        self.conv0_0 = nn.Sequential(
            conv(in_planes // 2, c // 4, 3, 2, 1),
            conv(c // 4, c // 2, 3, 2, 1)
        )
        self.conv0_1 = nn.Sequential(
            conv(in_planes // 2, c // 4, 3, 2, 1),
            conv(c // 4, c // 2, 3, 2, 1)
        )
        self.conv_block = nn.Sequential(
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c)
        )
        self.last_conv = nn.ConvTranspose2d(c, c // 2, 4, 2, 1)
        self.conv_clean_mask_0 = nn.Sequential(
            conv(c //2 + c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c//2),
            conv(c//2, 1)
        )
        self.conv_clean_mask_0 = nn.Sequential(
            conv(c //2 + c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c//2),
            conv(c//2, 1)
        )
    def forward(self, x, flow, scale):

        if scale != 1:
            x = F.interpolate(x, scale_factor=1. / scale, mode="bilinear", align_corners=False)
        if flow != None:
            flow = F.interpolate(flow, scale_factor=1. / scale, mode="bilinear", align_corners=False) * 1. / scale
            x = torch.cat((x, flow), 1)
        ds_img0 = x[:, :3]
        ds_img1 = x[:, 3:6]
        x_ds_img0 = self.conv0_0(ds_img0)
        x_ds_img1 = self.conv0_1(ds_img1)
        x = torch.cat((x_ds_img0, x_ds_img1), dim=1)
        x = self.conv_block(x) + x
        tmp = self.last_conv(x)
        tmp = F.interpolate(tmp, scale_factor=scale * 2, mode="bilinear", align_corners=False)
        flow = tmp[:, :4] * scale * 2
        mask = tmp[:, 4:5]

        mask_clean_mask_0 = self.conv_clean_mask_0(torch.cat([x, x_ds_img0], dim=1))
        mask_clean_mask_1 = self.conv_clean_mask_1(torch.cat([x, x_ds_img1], dim=1))

        return flow, mask, mask_clean_mask_0, mask_clean_mask_1


class Clean_Warp_IFNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.block0 = Student_Clean_Warp_IFBlock(6, c=240)
        self.block1 = Student_Clean_Warp_IFBlock(13 + 4, c=150)
        self.block2 = Student_Clean_Warp_IFBlock(13 + 4, c=90)
        self.block_tea = Student_Clean_Warp_IFBlock(16 + 4, c=90)
        self.contextnet = Contextnet()
        self.unet = Unet()

        def forward(self, x, scale=[4, 2, 1], timestep=0.5):
            img0 = x[:, :3]
            img1 = x[:, 3:6]
            gt = x[:, 6:]  # In inference time, gt is None
            flow_list = []
            merged = []
            mask_list = []
            warped_img0 = img0
            warped_img1 = img1
            flow = None
            loss_distill = 0
            stu = [self.block0, self.block1, self.block2]
            for i in range(3):
                if flow != None:
                    flow_d, mask_d, mask_clean_mask_0_d, mask_clean_mask_1_d = stu[i](
                        torch.cat((img0, img1, warped_img0, warped_img1, mask), 1), flow,
                        scale=scale[i])
                    flow = flow + flow_d
                    mask = mask + mask_d
                    mask_clean_mask_0 = mask_clean_mask_0 + mask_clean_mask_0_d
                    mask_clean_mask_1 = mask_clean_mask_1 + mask_clean_mask_1_d
                else:
                    flow, mask, mask_clean_mask_0, mask_clean_mask_1 = stu[i](torch.cat((img0, img1), 1), None,
                                                                              scale=scale[i])
                mask_list.append(torch.sigmoid(mask))
                flow_list.append(flow)

                warped_img0 = warp(img0, flow[:, :2])
                warped_img1 = warp(img1, flow[:, 2:4])
                clean_warped_img0 = mask_clean_mask_0 * warped_img0 + (1 - mask_clean_mask_0) * img1
                clean_warped_img1 = mask_clean_mask_1 * warped_img1 + (1 - mask_clean_mask_1) * img0
                merged_student = (clean_warped_img0, clean_warped_img1)
                merged.append(merged_student)
            if gt.shape[1] == 3:
                flow_d, mask_d = self.block_tea(torch.cat((img0, img1, warped_img0, warped_img1, mask, gt), 1), flow,
                                                scale=1)
                flow_teacher = flow + flow_d
                warped_img0_teacher = warp(img0, flow_teacher[:, :2])
                warped_img1_teacher = warp(img1, flow_teacher[:, 2:4])
                mask_teacher = torch.sigmoid(mask + mask_d)
                merged_teacher = warped_img0_teacher * mask_teacher + warped_img1_teacher * (1 - mask_teacher)
            else:
                flow_teacher = None
                merged_teacher = None
            for i in range(3):
                merged[i] = merged[i][0] * mask_list[i] + merged[i][1] * (1 - mask_list[i])
                if gt.shape[1] == 3:
                    loss_mask = ((merged[i] - gt).abs().mean(1, True) > (merged_teacher - gt).abs().mean(1,
                                                                                                         True) + 0.01).float().detach()
                    loss_distill += (
                            ((flow_teacher.detach() - flow_list[i]) ** 2).mean(1, True) ** 0.5 * loss_mask).mean()

            tmp = self.unet(img0, img1, warped_img0, warped_img1, mask, flow)
            res = tmp[:, :3] * 2 - 1
            merged[2] = torch.clamp(merged[2] + res, 0, 1)
            return flow_list, mask_list[2], merged, flow_teacher, merged_teacher, loss_distill


class Teacher_IFBlock(nn.Module):
    def __init__(self, in_planes=9, c=64):
        super().__init__()
        self.conv0_0 = nn.Sequential(
            conv(in_planes // 3, c // 6, 3, 2, 1),
            conv(c // 6, c // 3, 3, 2, 1)
        )
        self.conv0_1 = nn.Sequential(
            conv(in_planes // 3, c // 6, 3, 2, 1),
            conv(c // 6, c // 3, 3, 2, 1)
        )
        self.conv0_2 = nn.Sequential(
            conv(in_planes // 3, c // 6, 3, 2, 1),
            conv(c // 6, c // 3, 3, 2, 1)
        )
        self.conv_block = nn.Sequential(
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c)
        )
        self.last_conv = nn.ConvTranspose2d(c, c // 2, 4, 2, 1)
        self.teacher_conv_clean_mask_0 = nn.Sequential(
            conv(c //3 + c + c//3 , c),
            conv(c, c),
            conv(c, c),
            conv(c, c//3),
            conv(c//3, 1)
        )
        self.teacher_conv_clean_mask_1 = nn.Sequential(
            conv(c //3 + c + c//3 , c),
            conv(c, c),
            conv(c, c),
            conv(c, c//3),
            conv(c//3, 1)
        )

    def forward(self, x, flow, scale):

        if scale != 1:
            x = F.interpolate(x, scale_factor=1. / scale, mode="bilinear", align_corners=False)
        if flow != None:
            flow = F.interpolate(flow, scale_factor=1. / scale, mode="bilinear", align_corners=False) * 1. / scale
            x = torch.cat((x, flow), 1)
        ds_img0 = x[:, :3]
        ds_img1 = x[:, 3:6]
        ds_gt = x[:, 6:]
        x_ds_img0 = self.conv0_0(ds_img0)
        x_ds_img1 = self.conv0_1(ds_img1)
        x_ds_gt = self.conv0_2(ds_gt)
        x = torch.cat((x_ds_img0, x_ds_img1, x_ds_gt), dim=1)
        x = self.conv_block(x) + x
        tmp = self.last_conv(x)
        tmp = F.interpolate(tmp, scale_factor=scale * 2, mode="bilinear", align_corners=False)
        flow = tmp[:, :4] * scale * 2
        mask = tmp[:, 4:5]

        mask_clean_mask_0 = self.teacher_conv_clean_mask_0(torch.cat([x, x_ds_img0, x_ds_gt], dim=1))
        mask_clean_mask_1 = self.teacher_conv_clean_mask_1(torch.cat([x, x_ds_img1, x_ds_gt], dim=1))

        return flow, mask, mask_clean_mask_0, mask_clean_mask_1
