#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact.
"""
import argparse
import logging
import wandb
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    # Initialize W&B
    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact
    input_artifact = getattr(args, "input_artifact")
    logger.info(f"Downloading input artifact: {input_artifact}")
    artifact_path = run.use_artifact(input_artifact).file()
    logger.info(f"Downloaded input artifact:.")

    # Load data
    df = pd.read_csv(artifact_path)
    logger.info("Loaded data.")
    logger.info(f"Data shape: {df.shape}")

    # Clean data - Remove outliers
    min_price = getattr(args, "min_price")
    max_price = getattr(args, "max_price")
    df = df[(df["price"] >= min_price) & (df["price"] <= max_price)]
    logger.info(f"Cleaned data shape: {df.shape}")

    # Storing cleaned data
    output_artifact = getattr(args, "output_artifact")
    output_type = getattr(args, "output_type")
    output_description = getattr(args, "output_description")
    logger.info(f"Storing cleaned data to {output_artifact}")
    df.to_csv(output_artifact, index=False)
    artifact = wandb.Artifact(output_artifact, type=output_type, description=output_description)
    artifact.add_file(output_artifact)
    run.log_artifact(artifact)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")

    parser.add_argument(
        "--input_artifact", type=str, help="Fully-qualified artifact name for the input artifact", required=True
    )

    parser.add_argument(
        "--output_artifact", type=str, help="Name of Weights & Biases artifact to create", required=True
    )

    parser.add_argument(
        "--output_type", type=str, help="Step of the pipeline that will generate the output artifact", required=True
    )

    parser.add_argument(
        "--output_description", type=str, help="Description of the Artifact to be created", required=True
    )

    parser.add_argument(
        "--min_price",
        type=float,
        help="Exclude data points below a set value to remove unusually low values.",
        required=True,
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help="Exclude data points above a set value to remove unusually high values.",
        required=True,
    )

    args = parser.parse_args()

    go(args)
