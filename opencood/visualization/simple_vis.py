from matplotlib import pyplot as plt
import numpy as np

import opencood.visualization.simple_plot3d.canvas_3d as canvas_3d
import opencood.visualization.simple_plot3d.canvas_bev as canvas_bev

def visualize(
        pred_box_np, 
        gt_box_np, 
        pcd, 
        pc_range, 
        save_path, 
        method='3d', 
        vis_gt_box=True, 
        vis_pred_box=True, 
        left_hand=False, 
        vis_ref_car=False,
        uncertainty=None,
        refined_ref_pred_np=None,
        discovered_ref_pred_np=None,
        refcar_np=None,
        canvas_bg_color=(0, 0, 0)
        ):
    """
    Visualize the prediction, ground truth with point cloud together.
    They may be flipped in y axis. Since carla is left hand coordinate, while kitti is right hand.

    Parameters
    ----------
    pred_box_tensor : torch.Tensor
        (N, 8, 3) prediction.

    gt_tensor : torch.Tensor
        (N, 8, 3) groundtruth bbx

    pcd : torch.Tensor
        PointCloud, (N, 4).

    pc_range : list
        [xmin, ymin, zmin, xmax, ymax, zmax]

    save_path : str
        Save the visualization results to given path.

    dataset : BaseDataset
        opencood dataset object.

    method: str, 'bev' or '3d'

    """

    pc_range = [int(i) for i in pc_range]
    if isinstance(pcd, list):
        pcd_np = [x for x in pcd]
    else:
        pcd_np = pcd

    if vis_pred_box and pred_box_np is not None:
        #pred_box_np = pred_box_tensor.cpu().numpy()
        # pred_name = ['pred'] * pred_box_np.shape[0]
        pred_name = [''] * pred_box_np.shape[0]
        if uncertainty is not None:
            uncertainty_np = uncertainty.cpu().numpy()
            uncertainty_np = np.exp(uncertainty_np)
            d_a_square = 1.6**2 + 3.9**2
            
            if uncertainty_np.shape[1] == 3:
                uncertainty_np[:,:2] *= d_a_square
                uncertainty_np = np.sqrt(uncertainty_np) 
                # yaw angle is in radian, it's the same in g2o SE2's setting.

                pred_name = [f'x_u:{uncertainty_np[i,0]:.3f} y_u:{uncertainty_np[i,1]:.3f} a_u:{uncertainty_np[i,2]:.3f}' \
                                for i in range(uncertainty_np.shape[0])]

            elif uncertainty_np.shape[1] == 2:
                uncertainty_np[:,:2] *= d_a_square
                uncertainty_np = np.sqrt(uncertainty_np) # yaw angle is in radian

                pred_name = [f'x_u:{uncertainty_np[i,0]:.3f} y_u:{uncertainty_np[i,1]:3f}' \
                                for i in range(uncertainty_np.shape[0])]

            elif uncertainty_np.shape[1] == 7:
                uncertainty_np[:,:2] *= d_a_square
                uncertainty_np = np.sqrt(uncertainty_np) # yaw angle is in radian

                pred_name = [f'x_u:{uncertainty_np[i,0]:.3f} y_u:{uncertainty_np[i,1]:3f} a_u:{uncertainty_np[i,6]:3f}' \
                                for i in range(uncertainty_np.shape[0])]                    


    if vis_gt_box and gt_box_np is not None:
        #gt_box_np = gt_tensor.cpu().numpy()
        # gt_name = ['gt'] * gt_box_np.shape[0]
        gt_name = [''] * gt_box_np.shape[0] if gt_box_np is not None else []

    if method == 'bev':
        canvas = canvas_bev.Canvas_BEV_heading_right(
            canvas_shape=(
                (pc_range[4]-pc_range[1])*10, (pc_range[3]-pc_range[0])*10),
            canvas_x_range=(pc_range[0], pc_range[3]), 
            canvas_y_range=(pc_range[1], pc_range[4]),
            canvas_bg_color=canvas_bg_color,
            left_hand=left_hand
            ) 

        canvas_xy, valid_mask = canvas.get_canvas_coords(pcd_np) # Get Canvas Coords
        canvas.draw_canvas_points(
            canvas_xy[valid_mask],
            colors= (0, 0, 0)
            )
        # color_list = [(0, 206, 209),(255, 215,0)]
        # for i, pcd_np_t in enumerate(pcd_np[1:2]):
        #     canvas_xy, valid_mask = canvas.get_canvas_coords(pcd_np_t) # Get Canvas Coords
        #     canvas.draw_canvas_points(canvas_xy[valid_mask], colors=color_list[i-1]) # Only draw valid points
        box_line_thickness = 5
        if vis_gt_box and gt_box_np is not None:
            # canvas.draw_boxes(gt_box_np,colors=(0,255,0), texts=gt_name)
            canvas.draw_boxes(gt_box_np,colors=(0,255,0), texts=gt_name, box_line_thickness=box_line_thickness)
        
        if vis_pred_box and pred_box_np is not None:
            canvas.draw_boxes(pred_box_np, colors=(255,0,0), texts=pred_name, box_line_thickness=box_line_thickness)

        if refined_ref_pred_np is not None:
            refined_pred_name = [''] * refined_ref_pred_np.shape[0]
            canvas.draw_boxes(refined_ref_pred_np, colors=(255,165,0), texts=refined_pred_name, box_line_thickness=box_line_thickness)

        if discovered_ref_pred_np is not None:
            discovered_pred_name = [''] * discovered_ref_pred_np.shape[0]
            canvas.draw_boxes(discovered_ref_pred_np, colors=(61,133,198), texts=discovered_pred_name, box_line_thickness=box_line_thickness)


        if vis_pred_box and pred_box_np is not None:
            canvas.draw_boxes(pred_box_np, colors=(255,0,0), texts=pred_name, box_line_thickness=box_line_thickness)

        if vis_ref_car:
            canvas.draw_boxes(refcar_np, colors=(0,0,0), texts=[''], box_line_thickness=box_line_thickness)

    elif method == '3d':
        canvas = canvas_3d.Canvas_3D(left_hand=left_hand)
        canvas_xy, valid_mask = canvas.get_canvas_coords(pcd_np)
        canvas.draw_canvas_points(canvas_xy[valid_mask])
        
        if vis_gt_box and gt_box_np is not None:
            canvas.draw_boxes(gt_box_np,colors=(0,255,0), texts=gt_name)
        if vis_pred_box and pred_box_np is not None:
            canvas.draw_boxes(pred_box_np, colors=(255,0,0), texts=pred_name)
    else:
        raise(f"Not Completed for f{method} visualization.")

    plt.axis("off")

    plt.imshow(canvas.canvas)

    plt.tight_layout()
    plt.savefig(save_path, transparent=False, dpi=400, pad_inches=0.0)
    plt.clf()
    # print(save_path)



def visualize_data_split(
        pred_box_np, 
        pred2_box_np,
        gt_box_np, 
        ref_gt_box_np,
        refcar_bbox_np,
        pcd, 
        pc_range, 
        save_path, 
        method='3d', 
        vis_gt_box=True, 
        vis_pred_box=True, 
        left_hand=False, 
        vis_ref_car=False,
        uncertainty=None,
        canvas_bg_color=(0, 0, 0),
        put_dist=True,
        distance=999.
        ):
    """
    Visualize the prediction, ground truth with point cloud together.
    They may be flipped in y axis. Since carla is left hand coordinate, while kitti is right hand.

    Parameters
    ----------
    pred_box_tensor : torch.Tensor
        (N, 8, 3) prediction.

    gt_tensor : torch.Tensor
        (N, 8, 3) groundtruth bbx

    pcd : torch.Tensor
        PointCloud, (N, 4).

    pc_range : list
        [xmin, ymin, zmin, xmax, ymax, zmax]

    save_path : str
        Save the visualization results to given path.

    dataset : BaseDataset
        opencood dataset object.

    method: str, 'bev' or '3d'

    """

    pc_range = [int(i) for i in pc_range]
    if isinstance(pcd, list):
        pcd_np = [x for x in pcd]
    else:
        pcd_np = pcd

    if vis_pred_box and pred_box_np is not None:
        #pred_box_np = pred_box_tensor.cpu().numpy()
        # pred_name = ['pred'] * pred_box_np.shape[0]
        #pred_name = ['Model 1 pred'] * pred_box_np.shape[0]
        pred_name = [''] * pred_box_np.shape[0]

    if vis_pred_box and pred2_box_np is not None:
        #pred_box_np = pred_box_tensor.cpu().numpy()
        # pred_name = ['pred'] * pred_box_np.shape[0]
        #pred2_name = ['Model 2 pred'] * pred2_box_np.shape[0]
        pred2_name = [''] * pred2_box_np.shape[0]

    if vis_gt_box:
        #gt_box_np = gt_tensor.cpu().numpy()
        # gt_name = ['gt'] * gt_box_np.shape[0]
        #gt_name = ['GT'] * gt_box_np.shape[0] if gt_box_np is not None else []
        gt_name = [''] * gt_box_np.shape[0] if gt_box_np is not None else []

    ref_gt_name = [''] * ref_gt_box_np.shape[0] if ref_gt_box_np is not None else []

    if method == 'bev':
        canvas = canvas_bev.Canvas_BEV_heading_right(
            canvas_shape=(
                (pc_range[4]-pc_range[1])*10, (pc_range[3]-pc_range[0])*10),
            canvas_x_range=(pc_range[0], pc_range[3]), 
            canvas_y_range=(pc_range[1], pc_range[4]),
            canvas_bg_color=canvas_bg_color,
            left_hand=left_hand
            ) 

        canvas_xy, valid_mask = canvas.get_canvas_coords(pcd_np) # Get Canvas Coords
        canvas.draw_canvas_points(
            canvas_xy[valid_mask],
            colors= (0, 0, 0)
            )
        # color_list = [(0, 206, 209),(255, 215,0)]
        # for i, pcd_np_t in enumerate(pcd_np[1:2]):
        #     canvas_xy, valid_mask = canvas.get_canvas_coords(pcd_np_t) # Get Canvas Coords
        #     canvas.draw_canvas_points(canvas_xy[valid_mask], colors=color_list[i-1]) # Only draw valid points
        box_line_thickness = 5

        canvas.draw_boxes(
            ref_gt_box_np,
            colors=(77,175,74), 
            texts=ref_gt_name, 
            box_line_thickness=box_line_thickness
        )

        if vis_gt_box:
            # canvas.draw_boxes(gt_box_np,colors=(0,255,0), texts=gt_name)
            canvas.draw_boxes(
                gt_box_np,
                colors=(199,233,192), 
                texts=gt_name, 
                box_line_thickness=box_line_thickness,
                #box_text_size=0.2
            )
        
        if vis_pred_box and pred_box_np is not None:
            canvas.draw_boxes(pred_box_np, colors=(228,26,28), texts=pred_name, box_line_thickness=box_line_thickness)
        
        if vis_pred_box and pred2_box_np is not None:
            canvas.draw_boxes(pred2_box_np, colors=(55,126,184), texts=pred2_name, box_line_thickness=box_line_thickness)

        if vis_ref_car:
            canvas.draw_boxes(refcar_bbox_np, colors=(255,165,0), texts=[''], box_line_thickness=box_line_thickness)

    elif method == '3d':
        canvas = canvas_3d.Canvas_3D(left_hand=left_hand)
        canvas_xy, valid_mask = canvas.get_canvas_coords(pcd_np)
        canvas.draw_canvas_points(canvas_xy[valid_mask])
        
        if vis_gt_box:
            canvas.draw_boxes(gt_box_np,colors=(0,255,0), texts=gt_name)
        if vis_pred_box and pred_box_np is not None:
            canvas.draw_boxes(pred_box_np, colors=(255,0,0), texts=pred_name)
    else:
        raise(f"Not Completed for f{method} visualization.")

    if put_dist:
        canvas.draw_distance(distance)

    plt.axis("off")

    plt.imshow(canvas.canvas)

    plt.tight_layout()
    plt.savefig(save_path, transparent=False, dpi=400, pad_inches=0.0)
    plt.clf()
    # print(save_path)