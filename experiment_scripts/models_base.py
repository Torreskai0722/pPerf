#!/usr/bin/env python3
import os, subprocess, csv, pandas as pd, ast, time, sys
from itertools import product
from subprocess import TimeoutExpired
from nuscenes.nuscenes import NuScenes

# p_perf imports
sys.path.append('/mmdetection3d_ros2/perf_ws/src/p_perf')
from p_perf.pPerf_manager import pPerfConfigManager
from p_perf.post_process.timing_post import timing_processor
from p_perf.config.constant import image_models, lidar_models, seg_models

# Base nsys command
NSYS_BASE = ["nsys","profile","--trace=cuda,nvtx,cudnn","--backtrace=none","--force-overwrite","true"]
BAG_DIR = "/mmdetection3d_ros2/data/"
DATAROOT = "/mnt/nas/Nuscenes"
OVERWRITE = False
CONTINUE = True   # Set to True to continue from existing mapping file
LOGGING_DELAY = False

# Shared parameters
SCENES = ['2f0e54af35964a3fb347359836bec035_rainrate65', '2f0e54af35964a3fb347359836bec035_rainrate90', '2f0e54af35964a3fb347359836bec035_rainrate100']
DEPTHS = [-1]
IMG_QS = [1]
LIDAR_QS = [1]
PUB_RATES = [10]

# Experiment types
EXPERIMENTS = {
    "image_det-1": {
        "models": image_models,
        "queues": IMG_QS,
        "num_det": 1, "num_seg": 0,
        "csv_fields": ["run_index","scene","depth","image_model","image_queue","publishing_rate","status","start_time"]
    },
    "lidar_det-1": {
        "models": lidar_models,
        "queues": LIDAR_QS,
        "num_det": 1, "num_seg": 0,
        "csv_fields": ["run_index","scene","depth","lidar_model","lidar_queue","publishing_rate","status","start_time"]
    },
    "seg-1": {
        "models": seg_models,
        "queues": IMG_QS,  # no queue parameter
        "num_det": 0, "num_seg": 1,
        "csv_fields": ["run_index","scene","depth","seg_model","publishing_rate","status","start_time"]
    },
}


def run_exp(exp_name, exp_cfg):
    """Run the actual experiments with ros2+nsys and update mapping CSV."""
    print(f"\n=== RUNNING {exp_name.upper()} EXPERIMENTS ===")

    # Output dirs
    output_base = f"/mmdetection3d_ros2/outputs/{exp_name}_exps"
    os.makedirs(output_base, exist_ok=True)
    base_config_file = os.path.join(output_base, "base_config.yaml")

    # Config manager
    manager = pPerfConfigManager(base_config_file)
    manager.create_base_config(
        num_det_inferencers=exp_cfg["num_det"],
        num_seg_inferencers=exp_cfg["num_seg"],
        use_sim_time=True,
        logging_delay=LOGGING_DELAY
    )

    # Sweep setup
    if exp_name == "image_det-1":
        combinations = list(product(DEPTHS, exp_cfg["models"], SCENES, exp_cfg["queues"], PUB_RATES))
    elif exp_name == "lidar_det-1":
        combinations = list(product(DEPTHS, exp_cfg["models"], SCENES, exp_cfg["queues"], PUB_RATES))
    else:  # seg
        combinations = list(product(DEPTHS, exp_cfg["models"], SCENES, PUB_RATES))

    # Mapping CSV
    mapping_file = os.path.join(output_base, f"full_stack_mapping.csv")
    
    if OVERWRITE and not CONTINUE:
        # Complete overwrite - create new mapping file
        with open(mapping_file, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(exp_cfg["csv_fields"])
            for i, comb in enumerate(combinations):
                if exp_name == "image_det-1":
                    depth, model, scene, q, rate = comb
                    writer.writerow([i,scene,depth,model,q,rate,"pending",""])
                elif exp_name == "lidar_det-1":
                    depth, model, scene, q, rate = comb
                    writer.writerow([i,scene,depth,model,q,rate,"pending",""])
                else:  # seg - no queue parameter
                    depth, model, scene, rate = comb
                    writer.writerow([i,scene,depth,model,rate,"pending",""])
    elif CONTINUE and os.path.exists(mapping_file):
        # Continue mode - load existing mapping and add new combinations if any
        existing_df = pd.read_csv(mapping_file)
        
        # Create new combinations with indices starting from the last existing index + 1
        start_index = len(existing_df)
        new_combinations = []
        
        for i, comb in enumerate(combinations):
            # Check if this combination already exists
            combo_exists = False
            for _, row in existing_df.iterrows():
                if exp_name == "image_det-1":
                    depth, model, scene, q, rate = comb
                    if (row["scene"] == scene and row["depth"] == depth and 
                        str(row["image_model"]) == str(model) and row["image_queue"] == q and 
                        row["publishing_rate"] == rate):
                        combo_exists = True
                        break
                elif exp_name == "lidar_det-1":
                    depth, model, scene, q, rate = comb
                    if (row["scene"] == scene and row["depth"] == depth and 
                        str(row["lidar_model"]) == str(model) and row["lidar_queue"] == q and 
                        row["publishing_rate"] == rate):
                        combo_exists = True
                        break
                else:  # seg
                    depth, model, scene, rate = comb
                    if (row["scene"] == scene and row["depth"] == depth and 
                        str(row["seg_model"]) == str(model) and row["publishing_rate"] == rate):
                        combo_exists = True
                        break
            
            if not combo_exists:
                if exp_name == "image_det-1":
                    depth, model, scene, q, rate = comb
                    new_combinations.append([start_index + len(new_combinations),scene,depth,model,q,rate,"pending",""])
                elif exp_name == "lidar_det-1":
                    depth, model, scene, q, rate = comb
                    new_combinations.append([start_index + len(new_combinations),scene,depth,model,q,rate,"pending",""])
                else:  # seg
                    depth, model, scene, rate = comb
                    new_combinations.append([start_index + len(new_combinations),scene,depth,model,rate,"pending",""])
        

        # Append new combinations to existing file
        if new_combinations:
            with open(mapping_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                for combo in new_combinations:
                    writer.writerow(combo)
            print(f"Added {len(new_combinations)} new combinations to existing mapping file")
        else:
            print("No new combinations to add - all combinations already exist in mapping file")
    
    df = pd.read_csv(mapping_file)
    fail_log = os.path.join(output_base, "failures.log")
    with open(fail_log, "w") as flog:
        flog.write("Failed Runs\n===========\n")

    # Execution loop
    for i,row in df.iterrows():
        # Skip already completed experiments when continuing (both run_success and success)
        if row["status"] in ["run_success", "success"]:
            print(f"Skipping run {i} - already completed (status: {row['status']})")
            continue
            
        prefix = f"{output_base}/test_run_{i}"
        df.at[i,"status"]="pending"

        try:
            # Update sensor replay - use keyword arguments
            manager.update_sensor_replayer(
                bag_dir=BAG_DIR, 
                scene=row["scene"], 
                publishing_rate=row["publishing_rate"], 
                index=i
            )

            # Update inferencer(s)
            if exp_name=="image_det-1":
                manager.update_inferencer("det_inferencers",0,mode="image",model_name=row["image_model"],index=i,data_dir=output_base,image_queue=row["image_queue"])
            elif exp_name=="lidar_det-1":
                lidar_name,lidar_mode=ast.literal_eval(row["lidar_model"])
                manager.update_inferencer("det_inferencers",0,mode="lidar",model_name=lidar_name,index=i,data_dir=output_base,lidar_model_mode=lidar_mode,lidar_queue=row["lidar_queue"])
            else: # seg
                seg_name,seg_mode=ast.literal_eval(row["seg_model"])
                manager.update_inferencer("seg_inferencers",0,mode=seg_mode,model_name=seg_name,index=i,data_dir=output_base)

            config_file = base_config_file
            manager.save_config(config_file)

        except Exception as e:
            df.at[i,"status"]="config_error"
            open(fail_log,"a").write(f"Config fail {i}: {e}\n")
            df.to_csv(mapping_file,index=False)
            continue

        cmd = NSYS_BASE+["-o",prefix,"ros2","launch","p_perf","full_stack.launch.py",f"config_file:={config_file}"]
        start=time.time()
        try:
            subprocess.run(cmd,check=True,timeout=300)
            df.at[i,"status"]="run_success"; df.at[i,"start_time"]=start
        except TimeoutExpired as e:
            df.at[i,"status"]="timeout"; df.at[i,"start_time"]=start
        except Exception as e:
            df.at[i,"status"]="error"; df.at[i,"start_time"]=start
        finally:
            df.to_csv(mapping_file,index=False)

    print(f"✓ {exp_name} run stage complete → results in {output_base}")


def post_exp(exp_name, nusc):
    """Post-process successful runs (nsys export + timing_processor)."""
    output_base = f"/mmdetection3d_ros2/outputs/{exp_name}_exps"
    mapping_file = os.path.join(output_base, f"full_stack_mapping.csv")
    df = pd.read_csv(mapping_file)

    for i,row in df.iterrows():
        # Only process experiments that completed the run but haven't been post-processed yet
        if row["status"]!="run_success": continue
        prefix=f"{output_base}/test_run_{i}"
        raw=f"{prefix}.json"; rep=f"{prefix}.nsys-rep"
        if not os.path.exists(raw) and os.path.exists(rep):
            subprocess.run(["nsys","export","--type","json","--output",raw,rep],check=True)
        if os.path.exists(raw):
            tp=timing_processor(nusc,raw,output_base,i,scene=row["scene"],publish_mode="bag")
            tp.parse_json(); tp.generate_mapping(); tp.cleanup()
            os.remove(raw)
            
            # Mark as fully complete (run + post-processing done)
            df.at[i,"status"]="success"
            df.to_csv(mapping_file,index=False)

    print(f"✓ {exp_name} post-processing complete → results in {output_base}")


if __name__=="__main__":
    for exp_name, exp_cfg in EXPERIMENTS.items():
        run_exp(exp_name,exp_cfg)
    
    nusc=NuScenes(version='v1.0-trainval-rain',dataroot=DATAROOT)

    for exp_name in EXPERIMENTS.keys():
        post_exp(exp_name, nusc)
        
    print("\n=== ALL EXPERIMENTS COMPLETE ===")
