"""Run python -m src.demo --seconds 10, without starting an HTTP server."""
import argparse
import asyncio

from .main import Pipeline
from .schemas import StartRequest


async def demo(source: str, seconds: float, model: str | None, scenario: str | None = None) -> None:
    pipeline = Pipeline()
    consumer = asyncio.create_task(pipeline.consume_events())
    try:
        await pipeline.start(StartRequest(source=source, model_path=model, **({"scenario": scenario} if scenario else {})))
        await asyncio.sleep(seconds)
    finally:
        await pipeline.stop()
        try:
            await asyncio.wait_for(pipeline.events.join(), 20)
        finally:
            consumer.cancel()
            try:
                await consumer
            except asyncio.CancelledError:
                pass
    print(f"Demo complete: {pipeline.processed} frames, {len(pipeline.alerts)} alerts")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="synthetic")
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--model")
    parser.add_argument("--scenario", choices=["observe", "traffic", "interaction", "person_zone"])
    args = parser.parse_args()
    asyncio.run(demo(args.source, args.seconds, args.model, args.scenario))
