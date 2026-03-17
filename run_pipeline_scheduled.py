#!/usr/bin/env python3
"""
Wrapper to run the pipeline with cloud-agnostic scheduling support.
"""

import sys
import asyncio
import logging
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from scheduler.base import get_scheduler
import scrap
from ingestion.clean import clean_new_docs
from ingestion.chunk import chunk_new_docs
from ingestion.embed import embed_new_chunks

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_pipeline():
    """Core pipeline execution logic."""
    logger.info("=== PIPELINE START ===")
    
    logger.info("[Step 1] Scraping URLs...")
    urls = scrap.load_urls_from_file()
    if not urls:
        logger.warning("[stop] No URLs found in urls.txt or clgUrls.txt")
        return False

    try:
        asyncio.run(scrap.main(urls))
    except Exception as e:
        logger.error(f"[error] Scraping step failed: {e}", exc_info=True)
        return False
    
    logger.info("[Step 2] Cleaning docs...")
    try:
        clean_new_docs()
    except Exception as e:
        logger.error(f"[error] Cleaning step failed: {e}", exc_info=True)
        return False
    
    logger.info("[Step 3] Chunking docs...")
    try:
        chunk_new_docs()
    except Exception as e:
        logger.error(f"[error] Chunking step failed: {e}", exc_info=True)
        return False
    
    logger.info("[Step 4] Embedding and Storing in ChromaDB...")
    try:
        embed_new_chunks()
    except Exception as e:
        logger.error(f"[error] Embedding step failed: {e}", exc_info=True)
        return False
    
    logger.info("=== PIPELINE COMPLETE ===")
    return True


def main():
    """
    Main entry point with scheduler configuration.
    
    Environment variables:
    - SCHEDULER_TYPE: 'systemd' (default), 'apscheduler', 'kubernetes', 'serverless'
    - RUN_ONCE: Set to 'true' to run pipeline once and exit (for testing)
    """
    
    run_once = os.getenv("RUN_ONCE", "false").lower() == "true"
    scheduler_type = os.getenv("SCHEDULER_TYPE", "systemd")
    
    if run_once:
        logger.info("Running pipeline once (RUN_ONCE=true)")
        success = run_pipeline()
        sys.exit(0 if success else 1)
    
    # For scheduled execution
    logger.info(f"Initializing with scheduler: {scheduler_type}")
    
    try:
        scheduler = get_scheduler(scheduler_type)
        
        if scheduler_type == "apscheduler":
            logger.info("Starting APScheduler for recurring execution...")
            scheduler.schedule(run_pipeline, "monthly")
            scheduler.start()
            
            # Keep running
            try:
                logger.info("Scheduler running. Press Ctrl+C to stop.")
                while True:
                    asyncio.sleep(1)
            except KeyboardInterrupt:
                logger.info("Shutting down scheduler...")
                scheduler.stop()
        else:
            # For systemd, kubernetes, serverless: just show instructions
            logger.info(f"Using {scheduler_type} scheduler. Check deployment files for setup.")
            scheduler.schedule(run_pipeline, "monthly")
            scheduler.start()
    
    except Exception as e:
        logger.error(f"Scheduler initialization failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
