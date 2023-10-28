import tqdm
import argparse
import numpy as np
import torch
torch.multiprocessing.set_start_method("spawn", force=True)
from torch.utils import data
from networks.CDGNet import Res_Deeplab
from dataset.datasets import InferenceDataSet
import os
import torchvision.transforms as transforms
from copy import deepcopy
from utils.transforms import transform_parsing

from PIL import Image as PILImage

DATA_DIRECTORY = ''
DATA_LIST_PATH = ''
IGNORE_LABEL = 255
NUM_CLASSES = 20
SNAPSHOT_DIR = ''
INPUT_SIZE = (473,473)

# colour map
COLORS = [(0,0,0)
                # 0=background
                ,(128,0,0),(0,128,0),(128,128,0),(0,0,128),(128,0,128)
                # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                ,(0,128,128),(128,128,128),(64,0,0),(192,0,0),(64,128,0)
                # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                ,(192,128,0),(64,0,128),(192,0,128),(64,128,128),(192,128,128)
                # 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person
                ,(0,64,0),(128,64,0),(0,192,0),(128,192,0),(0,64,128)]
                # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
def get_lip_palette():
    palette = [
            0,0,0,
            243,50,30,
            242,112,31,
            93,167,220,
            150,215,228,
            7,113,189,
            26,208,244,
            158,209,156,
            192,215,63,
            242,234,12,
            239,175,27,
            30,156,243,
            ]        
    return palette               
def get_palette(num_cls):
  """ Returns the color map for visualizing the segmentation mask.

  Inputs:
    =num_cls=
      Number of classes.

  Returns:
      The color map.
  """
  n = num_cls
  palette = [0] * (n * 3)
  for j in range(0, n):
    lab = j
    palette[j * 3 + 0] = 0
    palette[j * 3 + 1] = 0
    palette[j * 3 + 2] = 0
    i = 0
    while lab:
      palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
      palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
      palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
      i += 1
      lab >>= 3
  return palette

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="CE2P Network")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--dataset", type=str, default='test',
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str,
                        help="Where restore model parameters from.")
    parser.add_argument("--gpu", type=str, default='0',
                        help="choose gpu device.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--output-path", type=str,
                        help="The output path.")
    parser.add_argument("--vis", type=str, default='yes',
                        help="Yes means to generate visualization results. No means generating nothing. ")

    return parser.parse_args()

def valid(model, valloader, input_size, num_samples, gpus):
    model.eval()

    parsing_preds = np.zeros((num_samples, input_size[0], input_size[1]),
                             dtype=np.uint8)

    scales = np.zeros((num_samples, 2), dtype=np.float32)
    centers = np.zeros((num_samples, 2), dtype=np.int32)

    idx = 0
    interp = torch.nn.Upsample(size=(input_size[0], input_size[1]), mode='bilinear', align_corners=True)
    with torch.no_grad():
        for index, batch in tqdm.tqdm(enumerate(valloader)):
            image, meta = batch
            num_images = image.size(0)
            # comment out by zjk
            # if index % 10 == 0:
            #     print('%d  processd' % (index * num_images))

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            scales[idx:idx + num_images, :] = s[:, :]
            centers[idx:idx + num_images, :] = c[:, :]
            #====================================================================================
            org_img = image.numpy()
            normal_img = org_img
            flipped_img = org_img[:,:,:,::-1]
            fused_img = np.concatenate( (normal_img,flipped_img), axis=0 )
            outputs = model( torch.from_numpy(fused_img).cuda())
            prediction = interp( outputs[0][-1].cpu()).data.numpy().transpose(0, 2, 3, 1) #N,H,W,C
            single_out = prediction[:num_images,:,:,:]
            single_out_flip = np.zeros( single_out.shape )
            single_out_tmp = prediction[num_images:, :,:,:]
            # for c in range(3):   # zjk
            #     single_out_flip[:,:, :, c] = single_out_tmp[:, :, :, c]
            single_out_flip[:,:, :, 0] = single_out_tmp[:, :, :, 0]
            single_out_flip[:,:, :, 1] = single_out_tmp[:, :, :, 1]
            single_out_flip[:,:, :, 2] = single_out_tmp[:, :, :, 2]
            single_out_flip[:,:, :, 11] = single_out_tmp[:, :, :, 11]
            
            single_out_flip[:, :, :, 3] = single_out_tmp[:, :, :, 4]
            single_out_flip[:, :, :, 4] = single_out_tmp[:, :, :, 3]
            single_out_flip[:, :, :, 5] = single_out_tmp[:, :, :, 6]
            single_out_flip[:, :, :, 6] = single_out_tmp[:, :, :, 5]
            single_out_flip[:, :, :, 7] = single_out_tmp[:, :, :, 8]
            single_out_flip[:, :, :, 8] = single_out_tmp[:, :, :, 7]
            single_out_flip[:, :, :, 9] = single_out_tmp[:, :, :, 10]
            single_out_flip[:, :, :, 10] = single_out_tmp[:, :, :, 9]
            single_out_flip           = single_out_flip[:, :, ::-1, :]  
            # Fuse two outputs
            single_out = ( single_out+single_out_flip ) / 2
            parsing_preds[idx:idx + num_images, :, :] = np.asarray(np.argmax(single_out, axis=3), dtype=np.uint8)

            idx += num_images

    parsing_preds = parsing_preds[:num_samples, :, :]


    return parsing_preds, scales, centers

def main():
    """Create the model and start the evaluation process."""
    args = get_arguments()

    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    gpus = [int(i) for i in args.gpu.split(',')]
    if len(gpus) != 1:
        raise KeyError(f"gpu number must be one during evaluating, but got {gpus}")

    h, w = map(int, args.input_size.split(','))
    
    input_size = (h, w)

    model = Res_Deeplab(num_classes=args.num_classes)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    lip_dataset = InferenceDataSet(args.data_dir, 'val', crop_size=input_size, transform=transform)

    num_samples = len(lip_dataset)

    valloader = data.DataLoader(lip_dataset, batch_size=args.batch_size * len(gpus),
                                shuffle=False, pin_memory=True)

    restore_from = args.restore_from
    print(f"The model is restored from: {restore_from}")

    state_dict = model.state_dict().copy()
    state_dict_old = torch.load(restore_from)

    for key, nkey in zip(state_dict_old.keys(), state_dict.keys()):
        if key != nkey:
            # remove the 'module.' in the 'key'
            state_dict[key[7:]] = deepcopy(state_dict_old[key])
        else:
            state_dict[key] = deepcopy(state_dict_old[key])

    model.load_state_dict(state_dict)

    model.eval()
    model.cuda()

    parsing_preds, scales, centers = valid(model, valloader, input_size, num_samples, len(gpus))

    #=================================================================
    save_dir = args.output_path

    save_lbl_dir = f'{save_dir}/Pred_parsing_results'
    print(f"The parsing results are saved to {save_lbl_dir}")
    
    if args.vis=='yes':
        save_vis_dir = f'{save_dir}/Pred_parsing_results_vis'
        if not os.path.exists( save_vis_dir ):
            os.makedirs( save_vis_dir )
        print(f"The visualization results are saved to {save_vis_dir}")

    palette = get_lip_palette()
    output_parsing = parsing_preds

    for i, im_lbl_name in tqdm.tqdm(enumerate(lip_dataset.files)):
        image_path = im_lbl_name['img']
        im_name = im_lbl_name['name']
        im_name = im_name.replace('.jpg', '.png')
        save_lbl_path = os.path.join(save_lbl_dir, im_name)
        os.makedirs(os.path.dirname(save_lbl_path), exist_ok=True)

        img = PILImage.open(image_path)

        w, h = img.size
        pred_out = output_parsing[i]
        s = scales[i]
        c = centers[i]
        pred = transform_parsing(pred_out, c, s, w, h, input_size)
        pred_lbl = PILImage.fromarray(pred)
        pred_lbl.save(save_lbl_path)

        # visualization (optional)
        if args.vis=='yes':
            save_vis_path = os.path.join(save_vis_dir, im_name)
            pred_vis = PILImage.fromarray(pred)
            pred_vis.putpalette(palette)
            pred_vis = pred_vis.convert("RGB")
            pred_vis.save(save_vis_path) 


if __name__ == '__main__':
    main()
