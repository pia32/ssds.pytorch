import os
import pickle
import os.path
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
from operator import itemgetter
from skimage.draw import polygon

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


VOC_CLASSES = ( '__background__', # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

VOC_CLASSES = ( '__background__', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20')

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))


class AnnotationTransform(object):

    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=True):
        self.class_to_ind = class_to_ind or dict(
            zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = np.empty((0,5)) 
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width
                #cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)

            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res = np.vstack((res,bndbox))  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class JACQUARDDetection(data.Dataset):

    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, root, image_sets, preproc=None, target_transform=AnnotationTransform(),
                 dataset_name='VOC0712'):
        self.root = root
        self.image_set = image_sets
        self.preproc = preproc
        self.target_transform = target_transform
        self.name = dataset_name
        self._annopath = os.path.join('%s', 'Annotations', '%s.xml')
        self._imgpath = os.path.join('%s', 'JPEGImages', '%s.jpg')
        self.ids = list()
        for (year, name) in image_sets:
            self._year = year
            rootpath = os.path.join(self.root, 'VOC' + year)
            for line in open(os.path.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                self.ids.append((rootpath, line.strip()))

    def __getitem__(self, index):
        img_id = self.ids[index]
        target = ET.parse(self._annopath % img_id).getroot()
        img = cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)
        height, width, _ = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target)


        if self.preproc is not None:
            img, target = self.preproc(img, target)
            #print(img.size())

                    # target = self.target_transform(target, width, height)
        #print(target.shape)

        return img, target

    def __len__(self):
        return len(self.ids)

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        # gt = self.target_transform(anno, 1, 1)
        # gt = self.target_transform(anno)
        # return img_id[1], gt
        if self.target_transform is not None:
            anno = self.target_transform(anno)
        return anno
        

    def pull_img_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        img = cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_transform(anno)
        height, width, _ = img.shape
        boxes = gt[:,:-1]
        labels = gt[:,-1]
        boxes[:, 0::2] /= width
        boxes[:, 1::2] /= height
        labels = np.expand_dims(labels,1)
        targets = np.hstack((boxes,labels))
        
        return img, targets

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        to_tensor = transforms.ToTensor()
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)

    def evaluate_detections(self, all_boxes, output_dir=None):
        """
        all_boxes is a list of length number-of-classes.
        Each list element is a list of length number-of-images.
        Each of those list elements is either an empty list []
        or a numpy array of detection.

        all_boxes[class][image] = [] or np.array of shape #dets x 5
        """
        self._write_voc_results_file(all_boxes)
        aps,map = self._do_python_eval(output_dir)
        return aps,map

    def _get_voc_results_file_template(self):
        filename = 'comp4_det_test' + '_{:s}.txt'
        filedir = os.path.join(
            self.root, 'results', 'VOC' + self._year, 'Main')
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        path = os.path.join(filedir, filename)
        return path

    def _write_voc_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(VOC_CLASSES):
            cls_ind = cls_ind 
            if cls == '__background__':
                continue
            print('Writing {} VOC results file'.format(cls))
            filename = self._get_voc_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.ids):
                    index = index[1]
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    for k in range(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                dets[k, 0] + 1, dets[k, 1] + 1,
                                dets[k, 2] + 1, dets[k, 3] + 1))

    def _do_python_eval(self, output_dir='output'):
        rootpath = os.path.join(self.root, 'VOC' + self._year)
        name = self.image_set[0][1]
        annopath = os.path.join(
                                rootpath,
                                'Annotations',
                                '{:s}.xml')
        imagesetfile = os.path.join(
                                rootpath,
                                'ImageSets',
                                'Main',
                                name+'.txt')
        cachedir = os.path.join(self.root, 'annotations_cache')
        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = True if int(self._year) < 2010 else False
        print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
        if output_dir is not None and not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        detDB = {}

        for i, cls in enumerate(VOC_CLASSES):
            if cls == '__background__':
                continue

            filename = self._get_voc_results_file_template().format(cls)

            detfile = filename.format(cls)
            with open(detfile, 'r') as f:
                lines = f.readlines()

            splitlines = [x.strip().split(' ') for x in lines]
            image_ids = [x[0] for x in splitlines]
            confidence = np.array([float(x[1]) for x in splitlines])
            BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

            for j in range(len(image_ids)):
                im_loc = image_ids[j]
                conf_loc = confidence[j]
                bb_loc = BB[j, :]

                if im_loc not in detDB:
                    detDB[im_loc] = []

                bb_entry = [conf_loc, int(cls), bb_loc[0], bb_loc[1], bb_loc[2], bb_loc[3]] #confidence, class, xmin, ymin, xmax, ymax
                detDB[im_loc].append(bb_entry)

        with open(imagesetfile, 'r') as f:
            lines = f.readlines()
        imagenames = [x.strip() for x in lines]

        total = 0
        suc = 0

        for im in imagenames:#foreach image
            if im not in detDB:
                print("No detections for image", im)
                continue

            bbDB = sorted(detDB[im], key=itemgetter(0), reverse=True)
            bestBB = bbDB[0]
            gtbbs = self.parse_rec(annopath.format(im))

            max_iou = self.calc_max_iou(bestBB, gtbbs)

            total += 1
            if max_iou > 0.25:
                suc += 1

            if total % 100 == 0:
                print(suc, total, suc/total)

        acc = suc / total
        print("FINAL ACCURACY", acc)
        return acc, acc

    def bb_to_corners(self, bb, angle_classes = 19):
        corners = np.zeros((4, 2))

        x = (bb[4] + bb[2]) / 2.0
        y = (bb[5] + bb[3]) / 2.0
        width = bb[4] - bb[2]
        height = bb[5] - bb[3]
        angle = (bb[1] - 1) / angle_classes * np.pi

        corners = np.zeros((4, 2));
        corners[0, 0] = -width / 2;
        corners[0, 1] = height / 2;
        corners[1, 0] = width / 2;
        corners[1, 1] = height / 2;
        corners[2, 0] = width / 2;
        corners[2, 1] = -height / 2;
        corners[3, 0] = -width / 2;
        corners[3, 1] = -height / 2;

        rot = [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
        corners = np.dot(corners, rot)

        corners = corners + np.array([x, y])

        return corners, angle

    def calc_max_iou(self, bb, gtbbs, visualize=False):
        max_iou = 0
        corners1, angle1 = self.bb_to_corners(bb)

        if visualize:
            img = np.zeros((1024, 1024, 3), np.uint8)
            self.cv2corners(img, corners1, color=(0, 255, 0))

        for i in range(len(gtbbs)):
            gtbb = gtbbs[i]
            gtbb = [1, int(gtbb['name']), gtbb['bbox'][0], gtbb['bbox'][1], gtbb['bbox'][2], gtbb['bbox'][3]]
            corners2, angle2 = self.bb_to_corners(gtbb)

            if visualize:
                self.cv2corners(img, corners2)

            if abs(angle2 - angle1) > np.pi / 6:
                continue

            iou = self.calc_iou(corners1, corners2)
            max_iou = max(iou, max_iou)

        if visualize:
            print(max_iou)
            cv2.imshow('result', img)
            cv2.waitKey(0)

        return max_iou

    def calc_iou(self, corners1, corners2):
        rr1, cc1 = polygon(corners1[:, 0], corners1[:, 1])
        rr2, cc2 = polygon(corners2[:, 0], corners2[:, 1])

        try:
            r_max = max(rr1.max(), rr2.max()) + 1
            c_max = max(cc1.max(), cc2.max()) + 1
        except:
            return 0

        canvas = np.zeros((r_max, c_max))
        canvas[rr1, cc1] += 1
        canvas[rr2, cc2] += 1
        union = np.sum(canvas > 0)
        if union == 0:
            return 0
        intersection = np.sum(canvas == 2)

        return intersection * 1.0 / union

    def cv2corners(self, img, corners, color=(255, 0, 0)):
        for i in range(4):
            nextI = (i + 1) % 4
            c1 = (int(corners[i, 0]), int(corners[i, 1]))
            c2 = (int(corners[nextI, 0]), int(corners[nextI, 1]))
            cv2.line(img, c1, c2, color, 3)

    def parse_rec(self, filename):
        """ Parse a PASCAL VOC xml file """
        tree = ET.parse(filename)
        objects = []
        for obj in tree.findall('object'):
            obj_struct = {}
            obj_struct['name'] = obj.find('name').text
            # obj_struct['pose'] = obj.find('pose').text
            obj_struct['truncated'] = int(obj.find('truncated').text)
            obj_struct['difficult'] = int(obj.find('difficult').text)
            bbox = obj.find('bndbox')
            obj_struct['bbox'] = [int(bbox.find('xmin').text),
                                  int(bbox.find('ymin').text),
                                  int(bbox.find('xmax').text),
                                  int(bbox.find('ymax').text)]
            objects.append(obj_struct)

        return objects

    def show(self, index):
        img, target = self.__getitem__(index)
        for obj in target:
            obj = obj.astype(np.int)
            cv2.rectangle(img, (obj[0], obj[1]), (obj[2], obj[3]), (255,0,0), 3)
        cv2.imwrite('./image.jpg', img)




## test
# if __name__ == '__main__':
#     ds = VOCDetection('../../../../../dataset/VOCdevkit/', [('2012', 'train')],
#             None, AnnotationTransform())
#     print(len(ds))
#     img, target = ds[0]
#     print(target)
#     ds.show(1)