import os
import click
from dotenv import load_dotenv
from data_loader import check_data, update_env_variable
from agent import run_graph

load_dotenv()


@click.group()
def cli():
    """LLM dataset analyzer"""
    pass


@cli.command()
@click.argument('prompt')
def ask(prompt: str):
    """Send a request to the model and get a response."""
    click.secho("User: " + prompt, fg='blue')
    click.secho("LLM: " + run_graph(prompt), fg='green')


@cli.command()
@click.option('--debug/--no-debug', default=False, help="Enabling debugging (displaying all steps).")
def setup(debug, think):
    """Application Settings."""
    update_env_variable('DEBUG', debug)
    click.echo(f"  DEBUG: {os.getenv('DEBUG')}")


@cli.command()
@click.option('--model', help="Name of model to use (for example, gpt-4, llama3, qwen3).")
@click.option('--base_url', help="Base URL for API requests.")
@click.option('--api_key', help="LLM provider API key.", hide_input=True)
def config(model, base_url, api_key):
    """Installing an OpenAi API compatible model."""
    if model:
        update_env_variable('MODEL_NAME', model)
    if base_url:
        update_env_variable('MODEL_BASE_URL', model)
    if base_url:
        update_env_variable('MODEL_API_KEY', api_key)
    click.echo("Current settings:")
    click.echo(f"  Model: {os.getenv('MODEL_NAME')}")
    click.echo(f"  Base_URL: {os.getenv('MODEL_BASE_URL')}")
    click.echo(f"  API_KEY: {os.getenv('MODEL_API_KEY')}")


@cli.command()
def load():
    """Uploading the dataset to the database."""
    check_data()
    click.echo(f"  DATASET_URL: {os.getenv('DATASET_URL')}")
    click.echo(f"  DATASET_NAME: {os.getenv('DATASET_NAME')}")
    click.echo(f"  DB_PATH: {os.getenv('DB_PATH')}")


if __name__ == '__main__':
    cli()
