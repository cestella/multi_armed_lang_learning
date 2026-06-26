"""CLI entry points for the language tutor."""

from __future__ import annotations

import click


@click.group()
def cli() -> None:
    """Adaptive conversational language tutor."""


@cli.command("tui")
@click.option("--language", default="it", type=click.Choice(["it", "es"]), help="Target language")
@click.option("--data-dir", default=".", help="Data directory for state and logs")
@click.option(
    "--config",
    "config_path",
    default=None,
    help="Path to config.yaml (default: auto-discover)",
)
def tui(language: str, data_dir: str, config_path: str | None) -> None:
    """Run the interactive TUI tutor."""
    import os
    from language_learning.config import load_config
    from language_learning.tui.app import TutorApp

    config = load_config(config_path)
    app = TutorApp(
        language=language,
        data_dir=os.path.abspath(data_dir),
        config=config,
    )
    app.run()


@cli.command("serve")
@click.option("--language", default="it", type=click.Choice(["it", "es"]), help="Target language")
@click.option("--data-dir", default=".", help="Data directory for state and logs")
@click.option(
    "--config",
    "config_path",
    default=None,
    help="Path to config.yaml (default: auto-discover)",
)
@click.option("--host", default="127.0.0.1", help="Bind host")
@click.option("--port", default=8080, type=int, help="Bind port")
def serve(language: str, data_dir: str, config_path: str | None, host: str, port: int) -> None:
    """Run the HTTP chat server with iMessage-style web UI."""
    import os
    import uvicorn
    from language_learning.config import load_config
    from language_learning.web.server import create_app

    config = load_config(config_path)
    app = create_app(config, language=language, data_dir=os.path.abspath(data_dir))
    print(f"Starting Language Tutor at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    cli()
