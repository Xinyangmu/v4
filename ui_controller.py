from argparse import ArgumentParser

import cv2
from mmengine.model import revert_sync_batchnorm

from mmseg.apis import inference_model, init_model, show_result_pyplot
from ui_v8 import Detect


class Detect_ROAD():
    def __init__(self):
        parser = ArgumentParser()
        parser.add_argument('--config',default=r"./configs/segformer/my_segformer_mit-b0_8xb2-160k_ade20k-512x512.py", help='Config file')
        parser.add_argument('--checkpoint',default=r"./tools/work_dirs/segformer_mit-b0_8xb2-160k_ade20k-512x512/iter_16000.pth", help='Checkpoint file')
        parser.add_argument('--out-file', default=None, help='Path to output file')
        parser.add_argument( '--device', default='cuda:0', help='Device used for inference')
        parser.add_argument( '--opacity', type=float, default=0.5, help='Opacity of painted segmentation map. In (0, 1] range.')
        parser.add_argument(  '--title', default='result', help='The image identifier.')
        self.args = parser.parse_args()
        self.model_seg = init_model(self.args.config, self.args.checkpoint, device=self.args.device)
        self.detects = Detect()
        if self.args.device == 'cpu':
            self.model_seg = revert_sync_batchnorm(self.model_seg)

    def detect(self,img0,callback= None):
        result = inference_model(self.model_seg, img0) # segmentation
        img0 = self.detects.detect(img0,None) #  detect
        vis_img = show_result_pyplot(
            self.model_seg,
            img0,
            result,
            title=self.args.title,
            opacity=self.args.opacity,
            draw_gt=False,
            show=False,
            out_file=self.args.out_file)

        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
        if callback is not None:
            callback(vis_img,"",False)
        return vis_img
if __name__=="__main__":
    dd = Detect_ROAD()
    xxx = dd.detect(cv2.imread("./demo/test.jpg"))
    cv2.imshow("xx",xxx)
    cv2.waitKey(0)
