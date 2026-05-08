from omegaconf import OmegaConf
from pprint import pprint
from utils import *
import json
import pickle

from agents import *
from prompt import *
import os




def run_single_sample(
    sample_index: int,
    sample: TableQASample,
    logger: logging.Logger,
):
    results = {
        "ids": sample.safe_get("ids"),
        "true_answer": sample.safe_get("answer"),
        "status": "Success",
    }
    
    try:
         
        logger.debug(f"=== Start Sample index={sample_index}, id={sample.safe_get('ids')} ===")
        
        agents["columner"].run(sample)
        
        
        agents["classifier"].get_classification(sample)
        
        agents["reasoner"].get_table_qa_answer(sample)
        
        results["first_result"] = sample.safe_get("first_qa_result_answer")
        
        agents["refiner"].refine_final_answer(sample)
        results["final_result"] = sample.safe_get("refine_history")
        
        
        

        logger.debug(f"=== Finished Sample index={sample_index}, id={sample.safe_get('ids')} ===")
        
    except Exception as e:
        results["status"] = "Failed"
        return results , sample
    
    return results , sample

def process_single_sample_and_save(
    idx,
    sample,
    config,
    write_lock,
    logger
):

    sample_id = str(sample.safe_get("ids", idx)) 
    
    context_dir = config.get("context_dir" , "./")
    result_file = config.get("result_file" , "./a.jsonl")
    
    try:
        out, ctx = run_single_sample(idx, sample, logger)
        
        LLM.show_tokens(True)

        pkl_path = os.path.join(context_dir, f"{sample_id}.pkl")
        
        try:
            with open(pkl_path, "wb") as pf:
                pickle.dump(ctx.to_dict(), pf)
        except Exception as e:
            logger.error(f"[ERROR] Failed to save pkl for id={sample_id}: {e}")

        with write_lock:
            try:
                with open(result_file, "a", encoding="utf8") as rf:
                    rf.write(json.dumps(out, ensure_ascii=False) + "\n")
            except Exception as e:
                logger.error(f"[ERROR] Failed to append result for id={sample_id}: {e}")
                
    except Exception as e:
        logger.exception(f"[FATAL ERROR] sample index={idx}, id={sample_id} FAILED: {e}")

def run_all_samples(config, samples, logger=None):
    num_workers = int(config.get("workers_num", 1))
    context_dir = config.get("context_dir")
    
    os.makedirs(context_dir, exist_ok=True)
    
    
    logger.info("Get_unprocessed_indices")
    filtered_indices = get_unprocessed_indices(samples, context_dir, logger)

    if not filtered_indices:
        logger.info("All samples have been processed (checked by IDs). Exiting.")
        return

    write_lock = threading.Lock()
    
    if num_workers <= 1:
        for idx in tqdm(filtered_indices, desc="Processing", unit="sample"):
            process_single_sample_and_save(
                idx, 
                samples[idx], 
                config,
                write_lock, 
                logger
            )
            
    else:
        with ThreadPoolExecutor(max_workers=config["workers_num"]) as executor:
            futures = []
            logger.info("Submitting tasks to thread pool...")
            for idx in filtered_indices:
                futures.append(
                    executor.submit(
                        process_single_sample_and_save,
                        idx,
                        samples[idx],
                        config,
                        write_lock,
                        logger
                    )
                )

            for f in tqdm(as_completed(futures), total=len(futures), desc="Processing", unit="sample" , ncols=100):
                try:
                    f.result()
                except Exception as e:
        
                    logger.error(f"Sample processing failed: {e}")

    if hasattr(LLM, 'show_tokens'):
        LLM.show_tokens(False)
    logger.info("All done.")



if __name__ == "__main__":
    config = OmegaConf.load("config.yaml")
    config = OmegaConf.to_container(config, resolve=True)
    
    
    config = setup_file_paths(config)
    logger = setup_logger(config["log_file"])
    
    logger.info("config info:\n" + json.dumps(config, indent=4, ensure_ascii=False))
    
    logger.info(f"Load date from {config['dataset_path']}")
    data_list = load_data(config)
    logger.info(f"Load {len(data_list)} samples from {config['dataset_path']}")

    LLM = LLMCaller(config , logger)

    
    
    prompt_config = PromptRegistry.get(config["dataset"])
    logger.info(f"Load  system prompt for {config['dataset_path']} : {prompt_config.DATASET_NAME}|{prompt_config.PROMPT_IDS}")
    
    agents = {
        "columner"  : ColumnAgent(LLM, logger , config , prompt_config ),
        "classifier": ClassifierAgent(LLM, logger , config ,prompt_config ),
        "reasoner"  : QaAgent(LLM, logger ,  config ,prompt_config),
        "refiner"   : RefineAgent(LLM, logger ,  config ,prompt_config , 3)
    }
    agents["refiner"].reasoning_agent = agents["reasoner"]
    
    
    
    run_all_samples(config  , data_list , logger)
    
    


