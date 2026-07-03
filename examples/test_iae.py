import open3d as o3d
import torch, os
from tqdm import tqdm
from torch.utils.data import DataLoader
import argparse
from learning3d.models import ImplicitAutoEncoder
from learning3d.data_utils import ClassificationData, ModelNet40Data

def display_open3d(template):
	template_ = o3d.geometry.PointCloud()
	template_.points = o3d.utility.Vector3dVector(template)
	# template_.paint_uniform_color([1, 0, 0])
	o3d.visualization.draw_geometries([template_])

def copy_parameters(model, pretrained_dict, verbose=True, classification=False):
    # ref: https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/3
    new_state_dict = {}
    for param_name in pretrained_dict:
        if 'encoder.dgcnn_encoder' in param_name:
            newname = param_name.replace('encoder.dgcnn_encoder', 'encoder')
            new_state_dict[newname] = pretrained_dict[param_name]
        elif 'encoder.foldnet_encoder' in param_name:
            newname = param_name.replace('encoder.foldnet_encoder', 'encoder')
            new_state_dict[newname] = pretrained_dict[param_name]
        elif 'module' in param_name:
            newname = param_name.replace('module', 'encoder')
            new_state_dict[newname] = pretrained_dict[param_name]
        else:
            new_state_dict[param_name] = pretrained_dict[param_name]
            
    if classification:
        new_state_dict = {'encoder.dgcnn_encoder.'+k: v for k, v in pretrained_dict.items()}

    pretrained_dict = new_state_dict
   
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                       k in model_dict and pretrained_dict[k].size() == model_dict[k].size()}

    if verbose:
        print('=' * 27)
        print('Restored Params and Shapes:')
        for k, v in pretrained_dict.items():
            print(k, ': ', v.size())
        print('=' * 68)
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model

def copy_parameters_ft(model, pretrained_dict, verbose=True):
    # ref: https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/3
    new_state_dict = {}
    for param_name in pretrained_dict:
        if 'encoder.dgcnn_encoder' in param_name:
            newname = param_name.replace('encoder.dgcnn_encoder.', '')
            new_state_dict[newname] = pretrained_dict[param_name]
        else:
            new_state_dict[param_name] = pretrained_dict[param_name]

    pretrained_dict = new_state_dict
   
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                       k in model_dict and pretrained_dict[k].size() == model_dict[k].size()}

    if verbose:
        print('=' * 27)
        print('Restored Params and Shapes:')
        for k, v in pretrained_dict.items():
            print(k, ': ', v.size())
        print('=' * 68)
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model

def test_one_epoch(device, model, test_loader, testset):
    model.eval()
    test_loss = 0.0
    pred  = 0.0
    count = 0
    for i, data in enumerate(tqdm(test_loader)):
        points, target = data
        target = target[:,0]

        points = points.to(device)
        target = target.to(device)

        output = model(None, points)
        loss_val = torch.nn.functional.nll_loss(
            torch.nn.functional.log_softmax(output, dim=1), target, size_average=False)
        print("Ground Truth Label: ", testset.get_shape(target[0].item()))
        print("Predicted Label:    ", testset.get_shape(torch.argmax(output[0]).item()))
        display_open3d(points.detach().cpu().numpy()[0])

        test_loss += loss_val.item()
        count += output.size(0)

        _, pred1 = output.max(dim=1)
        ag = (pred1 == target)
        am = ag.sum()
        pred += am.item()

    test_loss = float(test_loss)/count
    accuracy = float(pred)/count
    return test_loss, accuracy

def test(args, model, test_loader, testset):
    test_loss, test_accuracy = test_one_epoch(args.device, model, test_loader, testset)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

def options():
    parser = argparse.ArgumentParser(description='Point Cloud Registration')
    parser.add_argument('--dataset_path', type=str, default='ModelNet40',
                        metavar='PATH', help='path to the input dataset') # like '/path/to/ModelNet40'
    parser.add_argument('-b', '--batch_size', default=32, type=int,
                        metavar='N', help='mini-batch size (default: 32)')
    parser.add_argument('-j', '--workers', default=4, type=int,
                        metavar='N', help='number of data loading workers (default: 4)')

    # settings for on training
    parser.add_argument('--pretrained', default='learning3d/pretrained/exp_iae/modelnet40_trained.pt', type=str,
                        metavar='PATH', help='path to pretrained model file (default: null (no-use))')
    parser.add_argument('--device', default='cuda:0', type=str,
                        metavar='DEVICE', help='use CUDA if available')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = options()
    args.dataset_path = os.path.join(os.getcwd(), os.pardir, os.pardir, 'ModelNet40', 'ModelNet40')

    model = ImplicitAutoEncoder(classification=True)
    model_dict = torch.load(args.pretrained)['model_state_dict']
    model = copy_parameters(model, model_dict, verbose=True, classification=True)

    testset = ClassificationData(ModelNet40Data(train=False))
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.workers)

    test(args, model, test_loader, testset)