import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--track',  type=str, default="botsort.yaml" )  #botsort.yaml or bytetrack.yaml
    args = parser.parse_args()
    model = YOLO("yolov8m.pt")
    model.to('cuda')

    for result in model.track(source="./sample.mp4", show=True, stream=True, agnostic_nms=True,  tracker=args.track):
        pass
        
if __name__ == "__main__":
    main()