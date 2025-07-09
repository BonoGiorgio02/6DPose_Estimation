from evaluate_YOLO import get_YOLO
import time
from tqdm import tqdm
from utils.init import set_seed_inference
import torch.nn.functional as F
from data.CustomDataset import CustomDataset
import torch
from models.PoseEstimationPipeline import PoseEstimationPipeline
from PoseTrainer import PoseTrainer
from utils.pose_plot import plotPose
from models.PoseLossExtension import PoseLossExtension
from models.ADDMetricExtension import ADDMetricExtension
from data.CustomDataLoader import pointcloud_collate_fn
from torch.utils.data import DataLoader

def inference_extension(class_names=None, cam_K=None, device=torch.device("cpu"), path=None):
    set_seed_inference(42)

    dataset_root = "./datasets/linemod/DenseFusion/Linemod_preprocessed/"

    train_dataset = CustomDataset(dataset_root, split='train', device=device, cam_K=cam_K)
    image_mean, image_std = train_dataset.get_image_mean_std()
    val_dataset = CustomDataset(dataset_root, split='validation', device=device, cam_K = cam_K, img_mean = image_mean, img_std = image_std)
    test_dataset = CustomDataset(dataset_root, split='test', device=device, cam_K = cam_K, img_mean = image_mean, img_std = image_std)

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=pointcloud_collate_fn)

    MODELS_DIR = "./datasets/linemod/DenseFusion/Linemod_preprocessed/models"

    config = {
        "project_name": "pointnet",
        "experiment_name": "HiG",
        "batch_size": 16,
        "num_epochs": 17,
        "learning_rate": 4.0e-05,
        "weight_decay": 1e-5,
        "backbone": "resnet18",
        "hidden_dim": 512,
        "img_size": 224,
        "alpha": 1.0,
        "beta": 1.0,
        "add_threshold": 0.1,
        "symmetric_objects": ["10","11"],
        "name_saved_file": "HiG",
        "geometric_dims" : [64,128,256],
        "fusion_dim" : 128,
        "num_run_plotPose": 1
    }

    # load model
    object_detection_model = get_YOLO(path)
    model = PoseEstimationPipeline(fx=cam_K[0],fy=cam_K[4],cx=cam_K[2],cy=cam_K[5]).to(device)
    checkpoint = torch.load(f"./checkpoints/HiG_Resnet18_bs16.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    criterion = PoseLossExtension(class_names=class_names,device=device)

    # ADDMetric object used to call compute_add
    add_metric = ADDMetricExtension(
        model=model,
        class_names=class_names,
        test_loader=None,
        models_3D_dir=MODELS_DIR,
        symmetric_objects=config["symmetric_objects"],
        device=device,
        experiment=None,
        config=config
        )
    results = {}
    all_distances = []
    all_correct = []
    total_processed = 0
    total_skipped = 0
    not_found = 0

    total_time = 0

    wrong_detection = 0

    detection = {1: 0, 2: 0, 4: 0, 5: 0, 6: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0}

    for el in tqdm(test_loader):
        label, object_id = tuple(el["sample_id"].tolist()[0])
        # get image
        pathImage = f"./datasets/linemod/DenseFusion/Linemod_preprocessed/data/{label:02d}/rgb/{object_id:04d}.png"
        # get only most confident object
        start = time.time()
        result = object_detection_model.predict(pathImage, max_det=1)[0]
        if (len(result.boxes.xywh.tolist()) != 0):
            box = tuple(result.boxes.xywh.tolist()[0]) # [x_center, y_center, width, height]
            # store box in another format
            box = torch.tensor([box[0]-(box[2]/2), box[1]-(box[3]/2), box[2], box[3]])
            # crop image (resized and normalized)
            cropped_img = test_dataset.load_cropped_image(pathImage,tuple(box.tolist()))
            _, H, W = cropped_img.shape
            pad_H = round(H*1.0)
            pad_W = round(W*1.0)

            # compute symmetric padding: (left, right, top, bottom)
            pad_left = pad_W // 2 # pad 20% of cropped image
            pad_right = pad_W - pad_left
            pad_top = pad_H // 2
            pad_bottom = pad_H - pad_top

            padding = (round(pad_left), round(pad_right), round(pad_top), round(pad_bottom))
            # pad images by replicating the border pixels
            padded_img = F.pad(cropped_img, padding, mode='replicate')

            box = torch.stack([box]).to(device)
            padding = [torch.tensor([round(pad_left), round(pad_right), round(pad_top), round(pad_bottom)])]
            padding = torch.stack(padding).to(device)

            padded_img = torch.stack([padded_img]).to(device)

            # modify batch
            batch = el
            batch["cropped_img"] = padded_img
            batch["bbox_base"] = box
            batch["paddings"] = padding

            # pose
            gt_trans = batch['translation']
            gt_rot = batch['rotation']
            object_ids = batch['obj_id']
            obj_id = int(object_ids)
            obj_id = f"{obj_id:02d}"

            detected_object_id = result.summary()[0]["name"]
            detection[int(detected_object_id)] += 1
            if obj_id == detected_object_id:
                with torch.no_grad():
                    pixel_rotations_norm, pixel_translations, pixel_confidences = model(batch)
                    # compute loss
                    loss, r, t = criterion(pixel_rotations_norm, pixel_translations, pixel_confidences, gt_trans, gt_rot, object_ids)
                    total_time += time.time()-start

                    plotPose(pathImage, gt_trans[0], gt_rot[0], t[0], r[0], experiment=None, camera_intrinsics=cam_K)

                    pred_pose = (t[0], r[0])
                    gt_pose = (gt_trans[0], gt_rot[0])

                    distance, is_correct = add_metric.compute_add(pred_pose, gt_pose, obj_id, threshold=0.1*add_metric.dict_diameters[obj_id])

                    if not torch.isnan(distance):
                        all_distances.append(distance)
                        all_correct.append(is_correct)
                        total_processed += 1

                        if obj_id not in results:
                            results[obj_id] = {'distances': [], 'correct': []}

                        results[obj_id]['distances'].append(distance)
                        results[obj_id]['correct'].append(is_correct)
                    else:
                        total_skipped += 1
            else:
                wrong_detection += 1
        else:
            not_found += 1

    if len(all_distances) == 0:
        print("No valid objects found for evaluation.")
    else:
        # compute overall metrics
        print(f"Average time for one inference: {total_time/len(test_loader)} seconds")
        print(f"Wrong detections: {wrong_detection}/{len(test_loader)}")
        print(f"Total skipped: {total_skipped}/{len(test_loader)}")
        print(f"Not found objects: {not_found}/{len(test_loader)}")
        all_distances = torch.tensor(all_distances)
        all_correct = torch.tensor(all_correct)
        overall_add = torch.mean(all_distances)
        overall_accuracy = torch.mean(all_correct)

        print(f"\nOverall ADD: {overall_add:.4f}")
        print(f"Overall Accuracy: {overall_accuracy:.4f}")
        # print per-object results
        print("\nPer-object results:")
        for obj_id, obj_results in results.items():
            obj_add = torch.mean(torch.tensor(obj_results['distances']))
            obj_acc = torch.mean(torch.tensor(obj_results['correct']))
            num_samples = len(obj_results['distances'])
            print(f"Object {obj_id}: ADD={obj_add:.4f}, Acc={obj_acc:.4f}, Samples={num_samples}")