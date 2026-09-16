import json,time,cv2
from pathlib import Path
from offtheball.detection import PersonDetector
from offtheball.pipeline import analyze_video,create_run_directory
out=create_run_directory()
print(str(out),flush=True)
d=PersonDetector()
def progress(image,row,raw):
 if row["frame_index"] in (0,30,60,90,120,150,500,575,709,884):
  cv2.imencode(".jpg",image)[1].tofile(str(out/('preview-'+str(row["frame_index"])+'.jpg')))
 if row["frame_index"]%200==0:print('frame',row["frame_index"],'scene',row['scene_id'],flush=True)
out,summary=analyze_video("manvsmci.mp4",d,on_frame=progress,output_dir=out)
summary["tracker_type"]=type(d.model.predictor.trackers[0]).__name__
summary["version"]="0.2.0"
(out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps({"output":str(out),**summary},ensure_ascii=False),flush=True)

